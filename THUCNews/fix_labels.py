import os
import shutil

# --- 配置区 ---
# 确认你的数据集文件夹名字是否正确
DATASET_NAME = 'THUCNews'
# ----------------

# 构建文件路径
base_path = os.path.join(DATASET_NAME, 'data')
files_to_fix = ['train.txt', 'dev.txt', 'test.txt']

print("开始修正标签...")

for filename in files_to_fix:
    input_filepath = os.path.join(base_path, filename)
    # 创建一个临时文件名，用于写入修正后的内容
    output_filepath = os.path.join(base_path, filename + '.tmp')

    if not os.path.exists(input_filepath):
        print(f"警告：文件 {input_filepath} 不存在，已跳过。")
        continue

    try:
        lines_processed = 0
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label_str, text = parts
                    # 核心操作：将标签转换为整数，减1，再转回字符串
                    new_label = int(label_str) - 1

                    # 写入新行
                    outfile.write(f"{new_label}\t{text}\n")
                    lines_processed += 1

        # 用修正后的临时文件覆盖原文件
        shutil.move(output_filepath, input_filepath)
        print(f"文件 '{filename}' 修正完成！共处理 {lines_processed} 行。标签已从 1-14 映射到 0-13。")

    except Exception as e:
        print(f"处理文件 {filename} 时发生错误: {e}")
        # 如果出错，删除可能已创建的临时文件
        if os.path.exists(output_filepath):
            os.remove(output_filepath)

print("\n所有标签修正完毕！现在可以重新运行训练程序了。")