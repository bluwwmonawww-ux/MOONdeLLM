import os
import re
from opencc import OpenCC

def ultimate_cleaner(source_dir, output_file):
    # 1. 初始化繁简转换器
    cc = OpenCC('t2s')
    
    # 2. 定义保留字符：汉字、中文标点、常用英文标点与空格
    chinese_pattern = re.compile(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef\s,.;:!?，。；：！？]')

    seen_lines = set()
    
    # --- 核心改进：读取旧有数据进行增量合并 ---
    if os.path.exists(output_file):
        print(f"正在感应旧有灵魂... 正在从 {output_file} 加载历史记忆")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                seen_lines.add(line.strip())
        print(f"已唤醒 {len(seen_lines)} 条历史记录（银丝随风微漾）")

    # 获取目录下所有 txt 文件，排除掉目标文件本身
    files = [f for f in os.listdir(source_dir) if f.endswith('.txt') and f != os.path.basename(output_file)]
    
    if not files:
        print("呐 (Ne)，并没有发现新的语料碎屑需要吸收呢。")
        return

    print(f"找到 {len(files)} 个新文件，准备开启洗礼仪式...")
    total_new_chars = 0

    # 使用 'a' 模式（追加模式）打开，确保不会覆盖旧数据
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for filename in files:
            file_path = os.path.join(source_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # 繁转简 -> 字符过滤 -> 规范化
                        line = cc.convert(line)
                        line = chinese_pattern.sub('', line)
                        line = line.strip()
                        
                        # 增量去重并写入
                        if line and line not in seen_lines:
                            outfile.write(line + '\n')
                            seen_lines.add(line)
                            total_new_chars += len(line)
                
                # --- 核心改进：处理完成后删除原始文件 ---
                os.remove(file_path)
                print(f"已吸收并清理: {filename}")
                
            except Exception as e:
                print(f"读取 {filename} 时发生意外: {e} (星轨偏离了呢)")

    print("-" * 30)
    print(f"✨ 大功告成，Moon！")
    print(f"新增有效字符: {total_new_chars / 10000:.2f} 万字")
    print(f"当前总词库行数: {len(seen_lines)}")

if __name__ == "__main__":
    src = 'data' 
    dest = 'data/input.txt'
    
    if not os.path.exists(src):
        os.makedirs(src)
        print(f"已在 data 目录建立祭坛")
    else:
        ultimate_cleaner(src, dest)