import json
from collections import Counter

def extract_category_simple_keyword(json_file_path, categories):
    extracted_categories = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            # 一次性加载整个 JSON 文件
            json_data = json.load(f)

            # 假设 json_data 是一个 JSON 数组，遍历数组中的每个元素
            if isinstance(json_data, list):
                for data in json_data:
                    natural_language_topic = data
                    if not natural_language_topic:
                        extracted_categories.append("Other")
                        continue

                    assigned_category = "Other"
                    for category in categories:
                        if category.lower() in natural_language_topic.lower():
                            assigned_category = category
                            break
                    extracted_categories.append(assigned_category)
            # 如果 json_data 是一个 JSON 对象，但您期望的是一个数组，请根据实际情况调整处理逻辑
            elif isinstance(json_data, dict):
                print("Warning: JSON file seems to be a single JSON object, not an array as potentially expected.")
                # 您可能需要根据您的文件结构来处理这种情况，例如，如果这个 JSON 对象包含一个样本列表
                # 假设 JSON 对象包含一个名为 'samples' 的列表：
                # samples = json_data.get('samples', [])
                # for data in samples:
                #     ... (处理每个样本的代码，和上面数组的处理逻辑类似)
                pass # 或者您可能需要将整个 JSON 对象视为单个样本来处理，取决于文件结构
            else:
                print("Error: JSON file is not a list or a dictionary, unexpected format.")


        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            return ["Error - JSON Decode Failed"] # 返回错误标记，或者您可以选择抛出异常

    return extracted_categories

if __name__ == "__main__":
    json_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/topic.json"
    categories = [
        "Math",
        "Physics",
        "Chemistry",
        "Law",
        "Engineering",
        "Other",
        "Economics",
        "Health",
        "Psychology",
        "Business",
        "Biology",
        "Philosophy",
        "Computer Science",
        "Politics"
    ]

    extracted_categories = extract_category_simple_keyword(json_file, categories)

    print("提取的分类结果示例 (前 20 个):")
    for i in range(min(20, len(extracted_categories))):
        print(f"样本 {i+1}: {extracted_categories[i]}")

    output_file = "extracted_categories.txt"
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for category in extracted_categories:
            outfile.write(category + '\n')
    print(f"\n分类结果已保存到文件: {output_file}")

    # ---------------------  添加的统计类别数量和百分比的代码  ---------------------
    print("\n---------------------  类别统计  ---------------------")

    # 使用 Counter 统计每个类别的数量
    category_counts = Counter(extracted_categories)

    total_categories_extracted = len(extracted_categories)

    print("\n各类别数量和百分比:")
    for category, count in category_counts.items():
        percentage = (count / total_categories_extracted) * 100
        print(f"类别: {category}, 数量: {count}, 百分比: {percentage:.2f}%") #  %.2f  保留两位小数

    # 可选: 将类别统计结果保存到文件
    stats_output_file = "category_statistics.txt"
    with open(stats_output_file, 'w', encoding='utf-8') as stats_outfile:
        stats_outfile.write("类别统计结果:\n")
        stats_outfile.write("---------------------\n")
        for category, count in category_counts.items():
            percentage = (count / total_categories_extracted) * 100
            stats_outfile.write(f"类别: {category}, 数量: {count}, 百分比: {percentage:.2f}%\n")
    print(f"\n类别统计结果已保存到文件: {stats_output_file}")