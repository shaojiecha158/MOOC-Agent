import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# ================= 1. 路径配置 (适配你刚才的移动操作) =================
# 基座模型
BASE_MODEL_PATH = "/gemini/code/model/Qwen2___5-7B-Instruct"
# 你的微调模型 (你刚才 mv 过去的位置)
LORA_PATH = "/gemini/code/model/my_final_agent"

# ================= 2. 模型加载逻辑 (和你之前的一样，加了点防报错) =================
print(f"⏳ 正在加载基座模型: {BASE_MODEL_PATH} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
except Exception as e:
    print(f"❌ 致命错误：基座模型加载失败。\n原因: {e}")
    sys.exit()

print(f"🧠 正在挂载微调大脑 (LoRA): {LORA_PATH} ...")
try:
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA 挂载成功！Agent 已就绪。")
except Exception as e:
    print(
        f"❌ 致命错误：LoRA 加载失败。\n原因: {e}\n请检查 '/gemini/code/model/my_final_agent' 这个文件夹里有没有 adapter_model.safetensors 文件。")
    sys.exit()

model.eval()


# ================= 3. 对话生成逻辑 (和你之前的一样) =================
def generate_response(message, history):
    # 这里定义它的“人设”，非常重要
    system_prompt = (
        "你是一个智能教育规划师 MOOC-Agent。你的核心职责是根据用户的学习记录推荐课程。"
        "【重要规则】"
        "1. 如果用户输入的是学习记录、课程名或学科相关问题，请根据知识图谱规划路径。"
        "   - 有先修关系用【逻辑连贯】；"
        "   - 无关系用【内容推荐】。"
        "2. 如果用户问的是日常生活（如看病、天气）、专利代码或其他无关话题，请直接回答：“抱歉，我是专注于课程推荐的智能助手，无法回答该领域的通用问题。”"
        "3. 严禁在非课程推荐的场景下强行套用推荐模板。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.95,
            top_p=0.85,
            repetition_penalty=1.1,  # 惩罚复读机！强制它不准重复上一句的话
            do_sample=True,  # 确保开启采样
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ================= 4. 启动界面 (开启 share=True 最方便) =================
demo = gr.ChatInterface(
    fn=generate_response,
    title="🎓 MOOC-Agent 演示系统",
    description="基于 Qwen2.5 + MOOCCube 知识图谱微调。",
    examples=[
        ["用户已掌握: [C语言程序设计]. 用户目标: [数据结构]."],
        ["用户已掌握: [高等数学]. 用户目标: [人工智能]."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    # share=True 会生成一个公开链接，不用管端口映射也能访问
    demo.launch(server_name="0.0.0.0", share=True)