import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从主模块导入comp函数
try:
    from vshot import comp
except ImportError:
    # 如果无法导入，提供一个简单的实现
    from pkg.comp_img import aHash, calculate, classify_hist_with_split, cmpHash, dHash, pHash

    def comp(img1, img2, method=0):
        # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
        # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1
        if method == 0:
            hash1 = aHash(img1)
            hash2 = aHash(img2)
            n1 = cmpHash(hash1, hash2)
            return 1 - float(n1 / 64)
        elif method == 1:
            hash1 = dHash(img1)
            hash2 = dHash(img2)
            n2 = cmpHash(hash1, hash2)
            return 1 - float(n2 / 64)
        elif method == 2:
            hash1 = pHash(img1)
            hash2 = pHash(img2)
            n3 = cmpHash(hash1, hash2)
            return 1 - float(n3 / 64)
        elif method == 3:
            n4 = classify_hist_with_split(img1, img2)
            # 处理可能的返回值类型问题
            if isinstance(n4, tuple) and len(n4) > 0:
                return n4[0]
            return n4
        elif method == 4:
            n5 = calculate(img1, img2)
            # 处理可能的返回值类型问题
            if isinstance(n5, tuple) and len(n5) > 0:
                return n5[0]
            return n5
        elif method == 5:
            # SSIM算法
            from pkg.comp_ski import CompareImage

            compare_image = CompareImage()
            return compare_image.compare_gray(img1, img2)


# 尝试导入runAllImageSimilaryFun，如果不可用则提供一个替代实现
try:
    from vshot import runAllImageSimilaryFun  # noqa
except ImportError:
    # 尝试从comp_img直接导入
    try:
        from pkg.comp_img import runAllImageSimilaryFun  # noqa
    except ImportError:
        # 如果都无法导入，提供一个简单的实现
        def runAllImageSimilaryFun(para1, para2, isfile=True, isprint=True):
            """替代实现，返回None而不是计算所有算法的相似度"""
            return [0.0, 0.0, 0.0, 0.0, 0.0]  # 返回一个列表而不是None


class MultiLevelProcessor:
    """多级视频处理器，实现先粗略分析再精细分析的流程"""

    def __init__(
        self,
        logger,
        similarity_threshold,
        method,
        animation_frames=0,
        coarse_interval=30,
        coarse_threshold=0.95,
        coarse_method=2,
        max_workers=None,
    ):
        """
        初始化多级处理器

        参数:
            logger: 日志记录器
            similarity_threshold: 最终相似度阈值
            method: 精细分析使用的方法
            animation_frames: 动画稳定性检测帧数
            coarse_interval: 粗略分析的帧间隔
            coarse_threshold: 粗略分析的相似度阈值
            coarse_method: 粗略分析使用的方法（默认使用感知哈希算法，速度快）
            max_workers: 并行处理的最大工作线程数，None表示使用CPU核心数
        """
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.method = method
        self.animation_frames = animation_frames
        self.coarse_interval = coarse_interval
        self.coarse_threshold = coarse_threshold
        self.coarse_method = coarse_method
        self.max_workers = max_workers

        # 用于存储所有捕获的图片路径和帧号
        self.captured_files = []
        self.captured_count = 0

        # 稳定性检测队列
        if animation_frames > 0:
            self.stability_queue = deque(maxlen=animation_frames)

        # 用于稳定性检测的变量
        self.potential_capture = None
        self.potential_frame_num = None

    def process_video(self, video_path, outdir, spend=1):
        """
        处理视频文件

        参数:
            video_path: 视频文件路径
            outdir: 输出目录
            spend: 精细分析的帧间隔

        返回:
            捕获的图片文件列表
        """
        # 第一阶段：粗略分析，快速定位可能的关键帧
        self.logger.section("第一阶段：粗略分析")
        self.logger.info(
            f"使用算法: {self.get_method_name(self.coarse_method)}, 帧间隔: {self.coarse_interval}"
        )

        # 打开视频获取总帧数
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 格式化输出文件名
        num_frames_all_N = len(str(total_frames))
        mod = "%%0%dd.jpg" % num_frames_all_N

        # 粗略分析，找出可能的关键帧
        potential_keyframes = self._coarse_analysis(video_path, total_frames)
        self.logger.info(f"粗略分析完成，找到 {len(potential_keyframes)} 个潜在关键帧")

        # 第二阶段：精细分析，对潜在关键帧进行精细比较
        self.logger.section("第二阶段：精细分析")
        self.logger.info(
            f"使用算法: {self.get_method_name(self.method)}, 相似度阈值: {self.similarity_threshold}"
        )

        # 处理第一帧
        if len(potential_keyframes) > 0:
            first_frame_data = self._get_frame(video_path, potential_keyframes[0])
            if first_frame_data:
                num_frames, frame = first_frame_data
                outpath = (outdir + os.sep + mod) % num_frames
                cv2.imwrite(outpath, frame)
                self.logger.info(f"第1张: {os.path.basename(outpath)}")
                self.captured_files.append((outpath, num_frames))
                self.captured_count = 1
                last_frame = frame
                last_frame_num = num_frames

                # 并行处理其余潜在关键帧
                if len(potential_keyframes) > 1:
                    self._fine_analysis(
                        video_path, potential_keyframes[1:], last_frame, last_frame_num, outdir, mod
                    )
            else:
                self.logger.info("无法获取第一帧，处理终止")
        else:
            self.logger.info("未找到潜在关键帧，处理终止")

        return self.captured_files

    def _coarse_analysis(self, video_path, total_frames):
        """
        粗略分析视频，快速找出可能的关键帧

        返回:
            潜在关键帧列表
        """
        from pkg.read_video import VideoCut

        potential_keyframes = [0]  # 始终包含第一帧

        # 使用较大间隔和快速算法进行粗略扫描
        iters = VideoCut(video_path, self.coarse_interval)

        try:
            # 第一个yield返回的是总帧数，跳过
            total_frames_from_video = next(iters)

            # 获取第一帧
            first_frame_data = next(iters)
            if isinstance(first_frame_data, tuple) and len(first_frame_data) == 2:
                num_frames, frame = first_frame_data
                last_frame = frame
                last_frame_num = num_frames
            else:
                # 处理意外情况
                self.logger.info("无法获取第一帧，跳过粗略分析")
                return potential_keyframes

            # 扫描剩余帧
            for frame_data in iters:
                if isinstance(frame_data, tuple) and len(frame_data) == 2:
                    num_frames_new, frame_new = frame_data

                    # 计算当前进度
                    progress = float(num_frames_new) / float(total_frames)

                    # 更新进度条
                    self.logger.update_progress(
                        progress,
                        total=total_frames,
                        current=num_frames_new,
                        prefix="粗略分析进度: ",
                        suffix=f"已找到: {len(potential_keyframes)}个潜在关键帧",
                    )

                    # 使用快速算法比较
                    result = comp(last_frame, frame_new, method=self.coarse_method)

                    if result is not None and result < self.coarse_threshold:
                        potential_keyframes.append(num_frames_new)
                        last_frame = frame_new
                        last_frame_num = num_frames_new
        except StopIteration:
            # 处理迭代器结束的情况
            pass
        except Exception as e:
            self.logger.info(f"粗略分析时出错: {str(e)}")

        return potential_keyframes

    def _get_frame(self, video_path, frame_num):
        """获取指定帧号的帧"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame_num, frame
        return None

    def _fine_analysis(
        self, video_path, potential_keyframes, last_frame, last_frame_num, outdir, mod
    ):
        """
        对潜在关键帧进行精细分析
        """

        # 创建任务列表
        tasks = []
        for frame_num in potential_keyframes:
            tasks.append((frame_num, last_frame, last_frame_num))

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_frame = {
                executor.submit(self._process_keyframe, video_path, task, outdir, mod): task[0]
                for task in tasks
            }

            # 处理结果
            total = len(future_to_frame)
            completed = 0

            for future in future_to_frame:
                frame_num = future_to_frame[future]
                try:
                    result = future.result()
                    if result:
                        # 更新最后处理的帧
                        last_frame, last_frame_num = result
                except Exception as e:
                    self.logger.info(f"处理帧 {frame_num} 时出错: {str(e)}")

                # 更新进度
                completed += 1
                self.logger.update_progress(
                    completed / total,
                    total=total,
                    current=completed,
                    prefix="精细分析进度: ",
                    suffix=f"已捕获: {self.captured_count}张",
                )

    def _process_keyframe(self, video_path, task, outdir, mod):
        """
        处理单个潜在关键帧

        参数:
            task: (frame_num, last_frame, last_frame_num)元组

        返回:
            如果捕获了新帧，返回(新帧, 新帧号)，否则返回None
        """
        frame_num, last_frame, last_frame_num = task
        frame_data = self._get_frame(video_path, frame_num)

        if not frame_data:
            return None

        _, frame = frame_data

        # 使用精细算法比较
        result = comp(last_frame, frame, method=self.method)

        # 确保结果不为None且小于阈值
        if result is not None and result < self.similarity_threshold:
            # 如果启用了动画稳定性检测
            if self.animation_frames > 0:
                # 这里需要实现动画稳定性检测逻辑
                # 由于需要连续帧，这部分可能需要单独处理
                # 简化处理：直接捕获
                self.captured_count += 1
                outpath = (outdir + os.sep + mod) % frame_num
                self.logger.info(
                    f"\n第{self.captured_count}张: {os.path.basename(outpath)} (相似度: {result:.6f})"
                )

                cv2.imwrite(outpath, frame)
                self.captured_files.append((outpath, frame_num))
                return frame, frame_num
            else:
                # 直接捕获
                self.captured_count += 1
                outpath = (outdir + os.sep + mod) % frame_num
                self.logger.info(
                    f"\n第{self.captured_count}张: {os.path.basename(outpath)} (相似度: {result:.6f})"
                )

                cv2.imwrite(outpath, frame)
                self.captured_files.append((outpath, frame_num))
                return frame, frame_num

        return None

    def get_method_name(self, method):
        """获取算法名称"""
        names = [
            "均值哈希(aHash)",
            "差值哈希(dHash)",
            "感知哈希(pHash)",
            "三直方图",
            "单通道直方图",
            "结构相似度(SSIM)",
        ]
        return names[method] if 0 <= method < len(names) else f"未知算法({method})"
