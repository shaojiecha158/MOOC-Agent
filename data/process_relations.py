import json
import os
import random
from collections import defaultdict

# ================= 配置路径 =================
DATA_DIR = "data/MOOCCube"
ENTITIES_DIR = os.path.join(DATA_DIR, "entities")
RELATIONS_DIR = os.path.join(DATA_DIR, "relations")
OUTPUT_FILE = "data/mooc_agent_full_sft.json"


# ================= 1. 加载基础元数据 (从 entities 文件夹) =================
def load_metadata():
    print("Step 1: 加载课程元数据...")
    course_info = {}

    # 加载课程名称
    course_path = os.path.join(ENTITIES_DIR, "course.json")
    if os.path.exists(course_path):
        with open(course_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 清洗简介中的 HTML 标签（简单版）
                    desc = item.get('about', '') or item.get('name', '')
                    desc = desc.replace('<p>', '').replace('</p>', '').replace('&nbsp;', ' ')
                    course_info[item.get('id')] = {
                        "name": item.get('name'),
                        "desc": desc[:150]  # 截断，防止太长
                    }
                except:
                    continue

    # 加载概念名称 (用于在对话中显示概念名，而不是ID)
    concept_names = {}
    concept_path = os.path.join(ENTITIES_DIR, "concept.json")
    if os.path.exists(concept_path):
        with open(concept_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    concept_names[item.get('id')] = item.get('name')
                except:
                    continue

    return course_info, concept_names


# ================= 2. 加载关系数据 (从 relations 文件夹) =================
def load_relations():
    print("Step 2: 加载关系图谱...")

    # 1. 课程 -> 概念 (Course-Concept)
    c2k = defaultdict(list)
    with open(os.path.join(RELATIONS_DIR, "course-concept.json"), 'r', encoding='utf-8') as f:
        for line in f:
            # 兼容制表符或空格分隔
            parts = line.strip().split('\t') if '\t' in line else line.strip().split()
            if len(parts) >= 2:
                c2k[parts[0]].append(parts[1])  # CourseID -> ConceptID

    # 2. 概念先修关系 (Prerequisite)
    # 格式: A -> B (A是B的基础)
    pre_map = defaultdict(set)  # Post -> {Pre1, Pre2...}
    with open(os.path.join(RELATIONS_DIR, "prerequisite-dependency.json"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.strip().split()
            if len(parts) >= 2:
                pre_map[parts[1]].add(parts[0])  # 记录 B 需要 A

    return c2k, pre_map


# ================= 3. 生成逻辑解释 (The Logic Core) =================
def generate_reasoning(history_cids, target_cid, course_info, c2k, pre_map, concept_names):
    """
    核心函数：对比历史课程和目标课程，寻找知识关联，生成“人话”解释。
    """
    target_concepts = c2k.get(target_cid, [])
    history_concepts = set()
    for hc in history_cids:
        history_concepts.update(c2k.get(hc, []))

    reason_lines = []

    # 策略 A: 检查先修关系 (Strong Logic)
    # 检查目标课程的概念，是否依赖于历史课程里的概念
    found_prereq = False
    for tc in target_concepts:
        if tc in pre_map:  # 如果这个目标概念有先修要求
            needed_pres = pre_map[tc]
            # 交集：看历史里是否学过这些先修
            met_pres = needed_pres.intersection(history_concepts)
            if met_pres:
                # 找到了！构造句子
                pre_name = concept_names.get(list(met_pres)[0], "基础知识")
                target_c_name = concept_names.get(tc, "进阶知识")
                reason_lines.append(
                    f"**逻辑连贯**：你在之前的课程中已经接触了“{pre_name}”，这正是本课程核心概念“{target_c_name}”的先修基础，学习路径非常顺畅。")
                found_prereq = True
                break  # 找到一条最强的理由就够了

    # 策略 B: 检查内容重叠 (Semantic Similarity)
    # 如果没有强先修，看是否有概念重叠/相关
    if not found_prereq:
        overlap = set(target_concepts).intersection(history_concepts)
        if overlap:
            c_name = concept_names.get(list(overlap)[0], "相关知识")
            reason_lines.append(f"**兴趣延续**：该课程继续深入探讨了你感兴趣的“{c_name}”领域，有助于巩固你的知识体系。")
        else:
            # 兜底理由：使用课程简介
            desc = course_info[target_cid]['desc']
            reason_lines.append(f"**内容推荐**：该课程主要讲解：{desc}。")

    return "\n".join(reason_lines)


# ================= 4. 主程序：生成 ShareGPT 数据 =================
def main():
    course_info, concept_names = load_metadata()
    c2k, pre_map = load_relations()

    data = []

    # 加载用户选课记录
    user_file = os.path.join(RELATIONS_DIR, "user-course.json")
    print(f"Step 3: 处理用户数据 {user_file} ...")

    # user-course.json 格式: UserID \t CourseID \t Time...
    # 我们需要先聚合每个用户的课程列表
    user_histories = defaultdict(list)
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t') if '\t' in line else line.strip().split()
            if len(parts) >= 2:
                uid, cid = parts[0], parts[1]
                # 只有当课程在元数据里存在时才保留
                if cid in course_info:
                    user_histories[uid].append(cid)

    print(f"   聚合了 {len(user_histories)} 位用户的记录，开始生成对话...")

    system_prompt = "你是一个精通认知规律的AI教育顾问。请基于学习者的历史课程，推荐下一门课程，并从知识图谱的角度解释推荐理由（如先修关系、概念延续等）。"

    count = 0
    for uid, history in user_histories.items():
        # 过滤短序列
        if len(history) < 3: continue

        # 滑动窗口生成数据
        # Input: [A, B, C] -> Target: D
        # Context 长度设为 5
        context_len = 5

        # 为了避免数据量过大，每个用户只取最后 1-2 个样本
        # 也可以全取，看你的算力
        for i in range(max(1, len(history) - 2), len(history)):
            target_cid = history[i]
            input_cids = history[max(0, i - context_len): i]

            if not input_cids: continue

            # 构造 Input
            input_names = [f"《{course_info[c]['name']}》" for c in input_cids]
            user_text = f"我之前已经按顺序学习了以下课程：{', '.join(input_names)}。请推荐我的下一门课程。"

            # 构造 Output (包含推理)
            target_name = course_info[target_cid]['name']
            logic_reason = generate_reasoning(input_cids, target_cid, course_info, c2k, pre_map, concept_names)

            assistant_text = (
                f"基于你的学习轨迹，建议下一门课程学习《{target_name}》。\n\n"
                f"**推荐理由**：\n{logic_reason}"
            )

            data.append({
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "user", "value": user_text},
                    {"from": "assistant", "value": assistant_text}
                ]
            })
            count += 1

            if count % 5000 == 0:
                print(f"   ...已生成 {count} 条数据")

    # 乱序
    random.shuffle(data)

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成！共生成 {len(data)} 条微调数据，已保存至 {OUTPUT_FILE}")
    print("💡 关键点：生成的数据包含了基于 [prerequisite-dependency] 的显式逻辑推理。")


if __name__ == "__main__":
    main()