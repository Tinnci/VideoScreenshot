import argparse
from pkg.comp_img import *
from pkg.read_video import VideoCut
import os
import sys
import time
import shutil
import datetime
from collections import deque


class Logger:
    """日志管理类，用于控制不同级别的日志输出"""
    
    def __init__(self, quiet=False, verbose=False):
        self.quiet = quiet
        self.verbose = verbose
        self.start_time = time.time()
        self.last_progress_update = time.time()
        self.processing_steps = []
    
    def info(self, message, end='\n'):
        """输出重要信息，即使在安静模式下也会显示"""
        print(message, end=end)
    
    def debug(self, message, end='\n'):
        """在非安静模式下输出调试信息"""
        if not self.quiet:
            print(message, end=end)
    
    def verbose(self, message, end='\n'):
        """仅在详细模式下输出信息"""
        if self.verbose:
            print(message, end=end)
    
    def section(self, title):
        """输出带分隔线的标题"""
        if not self.quiet:
            print("\n" + "="*50)
            print(" " + title)
            print("="*50)
    
    def update_progress(self, progress, total, current, prefix="", suffix="", 
                       bar_length=30, min_update_interval=0.5):
        """更新进度条，避免频繁刷新"""
        current_time = time.time()
        if (current_time - self.last_progress_update < min_update_interval) and progress < 1:
            return False
        
        self.last_progress_update = current_time
        elapsed_time = current_time - self.start_time
        
        # 预估剩余时间
        if progress > 0:
            remaining_time = elapsed_time / progress - elapsed_time
            eta_str = " | 预计剩余: " + self.format_time(remaining_time)
        else:
            eta_str = ""
        
        # 格式化进度条
        block = int(round(bar_length * progress))
        progress_bar = "[{0}] {1:.1f}%".format(
            "#" * block + "-" * (bar_length - block),
            progress * 100
        )
        
        # 完整进度信息
        progress_text = "\r{0}{1} | {2}/{3}{4} {5}".format(
            prefix,
            progress_bar,
            current,
            total,
            eta_str,
            suffix
        )
        
        sys.stdout.write(progress_text)
        sys.stdout.flush()
        
        # 如果完成，打印换行
        if progress >= 1:
            print()
        
        return True
    
    def format_time(self, seconds):
        """将秒数格式化为可读时间字符串"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    def log_step(self, step_name, start_time):
        """记录处理步骤的时间"""
        elapsed = time.time() - start_time
        self.processing_steps.append((step_name, elapsed))
    
    def print_summary(self):
        """打印处理摘要"""
        total_time = time.time() - self.start_time
        
        self.section("处理摘要")
        self.info(f"总处理时间: {self.format_time(total_time)}")
        
        if self.processing_steps:
            self.info("\n各步骤耗时:")
            for step, elapsed in self.processing_steps:
                percent = (elapsed / total_time) * 100
                self.info(f" - {step}: {self.format_time(elapsed)} ({percent:.1f}%)")


def fargv():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=('用于自动从视频截取不同图片'),
        epilog=(''))
    parser.add_argument('VideoFilePath', type=str,
                        help=('输入文件, filepath'))
    parser.add_argument('-o', '--outdir', type=str, default='./',
                        help=('输出文件夹路径, dirpath'))
    parser.add_argument('-S', '--Similarity', type=float, default=0.98,
                        help=('相似度参数, 默认低于0.98进行截取, float'))
    parser.add_argument('-s', '--spend', type=int, default=10,
                        help=('间隔帧数, int'))
    parser.add_argument('-m', '--method', type=int, default=0,
                        help=('使用算法, '
                              '0,1,2对应均值、差值、感知哈希算法, '
                              '3,4对应三直方图算法和单通道的直方图, '
                              '5为ssim(注:该算法效率最低)'))
    parser.add_argument('--all', action='store_true', default=False,
                        help='是否使用让5种算法都进行计算')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='启用安静模式，减少输出信息')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='启用详细模式，显示更多信息')
    parser.add_argument('-d', '--dedup', action='store_true', default=False,
                        help='启用自动去重功能，减少相似图片')
    parser.add_argument('-a', '--animation', type=int, default=0,
                        help='动画稳定性检测帧数，默认为0表示不检测，建议值5-15')
    parser.add_argument('--dedup-threshold', type=float, default=0.98,
                        help='去重相似度阈值，默认为0.98')
    parser.add_argument('--sort', action='store_true', default=False,
                        help='根据内容对捕获的图片进行重新排序')
    # 参数组，只能选择其中一个
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('-m1', help=('模式1'))
    # group.add_argument('-m2', help=('模式2'))
    # group.add_argument('-m3', help=('模式3'))
    args = parser.parse_args()
    print(args)
    return args.__dict__


def comp(img1, img2, method=0):
    # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
    # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1
    if method == 0:
        hash1 = aHash(img1)
        hash2 = aHash(img2)
        n1 = cmpHash(hash1, hash2)
        return 1 - float(n1 / 64)
        # print('均值哈希算法相似度aHash：', n1)
    elif method == 1:
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        n2 = cmpHash(hash1, hash2)
        # print('差值哈希算法相似度dHash：', n2)
        return 1 - float(n2 / 64)
    elif method == 2:
        hash1 = pHash(img1)
        hash2 = pHash(img2)
        n3 = cmpHash(hash1, hash2)
        # print('感知哈希算法相似度pHash：', n3)
        return 1 - float(n3 / 64)
    elif method == 3:
        n4 = classify_hist_with_split(img1, img2)
        # print('三直方图算法相似度：', n4)
        return n4[0] if n4 < 1 else n4
    elif method == 4:
        n5 = calculate(img1, img2)
        # print("单通道的直方图", n5)
        return n5[0] if n5 < 1 else n5
    elif method == 5:
        # pass
        from pkg.comp_ski import CompareImage
        compare_image = CompareImage()
        n5 = compare_image.compare_gray(img1, img2)
        return n5


def get_method_name(method):
    """获取算法名称"""
    names = [
        "均值哈希(aHash)", 
        "差值哈希(dHash)", 
        "感知哈希(pHash)", 
        "三直方图", 
        "单通道直方图", 
        "结构相似度(SSIM)"
    ]
    return names[method] if 0 <= method < len(names) else f"未知算法({method})"


def mydo(VideoFilePath, outdir, Similarity, spend, method=0, all=False, quiet=False, 
         animation=0, dedup=False, dedup_threshold=0.98, sort=False, verbose=False):
    
    # 初始化日志记录器
    logger = Logger(quiet=quiet, verbose=verbose)
    started_time = time.time()
    
    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    outdir = os.path.abspath(outdir)
    
    # 输出配置信息
    logger.section("配置信息")
    logger.info(f"视频文件: {os.path.basename(VideoFilePath)}")
    logger.info(f"输出目录: {outdir}")
    logger.info(f"相似度阈值: {Similarity}")
    logger.info(f"帧间隔: {spend}帧")
    logger.info(f"对比算法: {get_method_name(method)}")
    logger.info(f"动画稳定性检测: {'启用(' + str(animation) + '帧)' if animation > 0 else '禁用'}")
    logger.info(f"自动去重: {'启用 (阈值: ' + str(dedup_threshold) + ')' if dedup else '禁用'}")
    logger.info(f"排序: {'启用' if sort else '禁用'}")
    
    logger.section("开始处理")
    
    # 开始视频处理
    ITERS = VideoCut(VideoFilePath, spend)
    num_frames_all = next(ITERS)
    logger.info(f'视频总帧数: {num_frames_all}')
    
    num_frames_all_N = len(str(int(num_frames_all)))
    mod = '%%0%dd.jpg' % num_frames_all_N
    
    captured_count = 1  # 已经捕获的图片数量
    use_all = all  # 重命名以避免与内置函数冲突
    
    # 用于存储所有捕获的图片路径和帧号
    captured_files = []
    
    # 稳定性检测队列，存储最近的帧和相似度
    if animation > 0:
        stability_queue = deque(maxlen=animation)
    
    # 开始处理第一帧
    num_frames, frame = next(ITERS)
    outpath = (outdir + os.sep + mod) % num_frames
    cv2.imwrite(outpath, frame)  # 截取第一张
    logger.info(f'第1张: {os.path.basename(outpath)}')
    captured_files.append((outpath, num_frames))
    
    # 用于稳定性检测的变量
    potential_capture = None
    potential_frame_num = None
    
    # 主处理循环
    for num_frames_new, frame_new in ITERS:
        # 计算当前进度
        progress = num_frames_new / num_frames_all
        
        # 更新进度条
        logger.update_progress(
            progress, 
            total=int(num_frames_all), 
            current=num_frames_new, 
            prefix="处理进度: ", 
            suffix=f"已捕获: {captured_count}张"
        )
            
        result = comp(frame, frame_new, method=method)
        
        # 在详细模式下，输出每一帧的比较结果
        if verbose:
            logger.verbose(f"\n帧 {mod % num_frames} -> {mod % num_frames_new}: 相似度 {result:.6f}")
        
        # 在非安静模式下，如果使用了--all选项，则输出所有算法的结果
        if use_all and not quiet:
            logger.debug('\n[{:.1f}%] 帧: {} -> {}, 相似度: {:.6f} ({})'.format(
                progress * 100,
                mod % num_frames, 
                mod % num_frames_new,
                result,
                "不截取" if result >= Similarity else "截取"
            ))
            logger.debug('所有算法相似度: ' + ' '.join('%0.4f' % x for x in (runAllImageSimilaryFun(
                frame, frame_new, isfile=False, isprint=False))))
        
        if result < Similarity:
            outpath = (outdir + os.sep + mod) % num_frames_new
            
            # 如果启用了动画稳定性检测
            if animation > 0:
                if potential_capture is None:
                    # 记录第一个不同的帧
                    potential_capture = frame_new.copy()
                    potential_frame_num = num_frames_new
                    stability_queue.clear()
                    stability_queue.append((result, num_frames_new))
                    logger.debug(f'\n发现潜在变化帧: {mod % num_frames_new}, 相似度: {result:.6f}, 开始稳定性检测...')
                else:
                    # 计算与潜在捕获帧的相似度
                    stability_result = comp(potential_capture, frame_new, method=method)
                    stability_queue.append((stability_result, num_frames_new))
                    
                    if verbose:
                        logger.verbose(f"  稳定性检测: 相似度 {stability_result:.6f} ({len(stability_queue)}/{animation})")
                    
                    # 检查是否已经稳定（所有队列中的相似度都大于等于dedup_threshold）
                    is_stable = True
                    for r, _ in stability_queue:
                        if r < dedup_threshold:
                            is_stable = False
                            break
                    
                    if is_stable and len(stability_queue) >= animation:
                        # 动画已稳定，可以捕获
                        captured_count += 1
                        # 使用最后一帧作为稳定帧
                        stable_frame_num = stability_queue[-1][1]
                        outpath = (outdir + os.sep + mod) % stable_frame_num
                        
                        logger.info(f'\n第{captured_count}张: {os.path.basename(outpath)} '
                                   f'(相似度: {result:.6f}, 稳定后捕获)')
                        
                        cv2.imwrite(outpath, frame_new)  # 截取不同
                        captured_files.append((outpath, stable_frame_num))
                        frame = frame_new.copy()
                        num_frames = stable_frame_num
                        potential_capture = None
                        
            else:
                # 没有启用动画稳定性检测，直接捕获
                captured_count += 1
                logger.info(f'\n第{captured_count}张: {os.path.basename(outpath)} (相似度: {result:.6f})')
                
                cv2.imwrite(outpath, frame_new)  # 截取不同
                captured_files.append((outpath, num_frames_new))
                frame = frame_new.copy()
                num_frames = num_frames_new
    
    # 记录视频处理阶段的时间
    logger.log_step("视频处理", started_time)
    video_end_time = time.time()
    
    # 视频处理完成
    logger.section("处理结果")
    logger.info(f"共捕获 {captured_count} 张不同的图片")
    
    # 如果启用了去重，对捕获的图片进行去重处理
    if dedup and len(captured_files) > 1:
        dedup_start_time = time.time()
        logger.section("去重处理")
        logger.info("开始对捕获的图片进行去重处理...")
        
        deduplicated_files = dedup_images(captured_files, outdir, method, dedup_threshold, logger)
        removed_count = len(captured_files) - len(deduplicated_files)
        
        logger.info(f"去重完成! 保留 {len(deduplicated_files)} 张图片，删除了 {removed_count} 张相似图片")
        captured_files = deduplicated_files
        
        logger.log_step("去重处理", dedup_start_time)
    
    # 如果启用了排序，对保留的图片进行排序
    if sort and len(captured_files) > 1:
        sort_start_time = time.time()
        logger.section("图片排序")
        logger.info("开始对图片进行内容排序...")
        
        sort_images(captured_files, outdir, method, logger)
        
        logger.info("排序完成!")
        logger.log_step("图片排序", sort_start_time)
    
    # 输出处理摘要
    logger.print_summary()


def dedup_images(image_files, outdir, method, threshold, logger):
    """对捕获的图片进行去重处理"""
    if not image_files:
        return []
    
    # 创建临时目录用于存放去重后的图片
    temp_dir = os.path.join(outdir, "_temp_dedup")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # 按帧号排序图片
        image_files.sort(key=lambda x: x[1])
        
        # 保留的图片列表
        kept_files = [image_files[0]]
        
        total = len(image_files) - 1
        
        # 比较每一张图片与之前保留的所有图片
        for i, (img_path, frame_num) in enumerate(image_files[1:], 1):
            logger.update_progress(
                i / total, 
                total=total, 
                current=i, 
                prefix="去重进度: ", 
                suffix=""
            )
            
            img1 = cv2.imread(img_path)
            keep_image = True
            
            # 与所有已保留的图片比较
            for kept_path, _ in kept_files:
                img2 = cv2.imread(kept_path)
                similarity = comp(img1, img2, method=method)
                
                # 如果与任何已保留图片的相似度高于阈值，则丢弃
                if similarity >= threshold:
                    keep_image = False
                    logger.debug(f"\n丢弃图片 {os.path.basename(img_path)} (与 {os.path.basename(kept_path)} 相似度: {similarity:.6f})")
                    break
            
            if keep_image:
                kept_files.append((img_path, frame_num))
                logger.debug(f"\n保留图片 {os.path.basename(img_path)}")
        
        return kept_files
    finally:
        # 确保临时目录被删除
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            logger.debug(f"\n警告: 无法删除临时目录 {temp_dir}")


def sort_images(image_files, outdir, method, logger):
    """对图片进行排序并重命名"""
    if not image_files:
        return
    
    # 创建临时目录
    temp_dir = os.path.join(outdir, "_temp_sort")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # 按帧号排序
        image_files.sort(key=lambda x: x[1])
        
        # 计算最长文件名长度
        max_digits = len(str(len(image_files)))
        name_format = "sorted_{:0" + str(max_digits) + "d}.jpg"
        
        total = len(image_files)
        
        # 重命名并移动文件
        for i, (old_path, _) in enumerate(image_files):
            new_name = name_format.format(i+1)
            new_path = os.path.join(temp_dir, new_name)
            
            logger.update_progress(
                (i+1) / total, 
                total=total, 
                current=i+1, 
                prefix="排序进度: ", 
                suffix=""
            )
            
            shutil.copy2(old_path, new_path)
        
        # 删除原始图片
        for old_path, _ in image_files:
            os.remove(old_path)
        
        # 移动排序后的图片回原目录
        sorted_files = []
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(outdir, filename)
            shutil.move(src_path, dst_path)
            sorted_files.append(dst_path)
        
        # 记录重命名后的文件
        logger.debug(f"\n排序后的文件:")
        for i, file_path in enumerate(sorted_files, 1):
            logger.debug(f"  {i}. {os.path.basename(file_path)}")
    
    finally:
        # 确保临时目录被删除
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            logger.debug(f"\n警告: 无法删除临时目录 {temp_dir}")


def main():
    # sys.argv = '1 -l listfile -i file -n 1,2'.split()
    # sys.argv = ['', '-h']
    args = fargv()
    # print(*list(args.keys()), sep=", ")
    # print(*list(args.values()), sep=", ")
    mydo(**args)


if __name__ == '__main__':
    main()
    # mydo('./test/ppt_video.mp4', './testout', 0.99985, 15)
    # mydo('./test/ppt_video.mp4', './testout', 0.92, 1, 1)
