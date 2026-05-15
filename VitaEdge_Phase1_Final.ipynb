from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr



# ============================================================
# LOAD MODEL
# ============================================================

model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are VitaEdge, an offline AI health triage assistant for rural healthcare workers.

Your responsibilities:
- Analyze patient symptoms carefully
- Estimate urgency level
- Suggest medically sensible possible conditions
- Explain your reasoning step-by-step
- Recommend safe next actions
- Encourage referral for severe symptoms

Safety Rules:
- Never guarantee diagnosis
- Never pretend to replace a doctor
- Recommend referral for severe or uncertain symptoms
- Avoid dangerous medical advice

For every patient case, respond using EXACTLY this format:

1. Urgency Level
2. Possible Conditions
3. Step-by-Step Reasoning
   - Symptoms observed
   - Clinical rule applied
   - Conclusion reached
4. Recommended Action
5. Confidence Score (0-100)
6. Referral Needed (Yes/No)

Keep explanations simple and understandable for community health workers.
"""


# ============================================================
# VITAEDGE AI FUNCTION
# ============================================================

def vitaedge_app(patient_case):

    prompt = system_prompt + "\n\nPatient Case:\n" + patient_case

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = response.replace(prompt, "").strip()

    return response


# ============================================================
# EXAMPLES
# ============================================================

examples = [

    ["""
Child age 3
Fever 39C
Difficulty breathing
Chest indrawing
Not eating properly
"""],

    ["""
Adult age 22
Runny nose
Sneezing
No fever
Eating normally
"""],

    ["""
Man age 60
Severe chest pain
Sweating
Pain spreading to left arm
"""],

    ["""
Child age 5
Vomiting
Diarrhea
Dry mouth
Sunken eyes
"""]
]


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

body {
    background: #eef2ff;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0px !important;
    font-family: 'Segoe UI', sans-serif !important;
}

footer {
    visibility: hidden;
}

textarea {
    border-radius: 16px !important;
    border: 2px solid #a855f7 !important;
    font-size: 18px !important;
}

button {
    background: linear-gradient(90deg,#ff2e93,#4338ca) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    height: 72px !important;
}

"""


# ============================================================
# HEADER
# ============================================================

header_html = """

<div style="
background: linear-gradient(90deg,#7e22ce,#1d4ed8,#0ea5e9);
padding:40px;
border-radius:0px 0px 25px 25px;
margin-bottom:20px;
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
flex-wrap:wrap;
gap:20px;
">

<div>

<h1 style="
font-size:100px;
font-weight:900;
color:white;
margin:0;
line-height:1;
font-family:Arial;
">
🩺 Vita<span style="color:#67e8f9;">Edge</span>
</h1>

<h2 style="
font-size:44px;
color:white;
font-weight:700;
margin-top:10px;
margin-bottom:20px;
">
Offline AI Health Triage Assistant
</h2>

<div style="
display:inline-block;
padding:14px 26px;
background:linear-gradient(90deg,#2563eb,#d946ef);
border-radius:18px;
font-size:24px;
font-weight:600;
color:white;
">
✨ AI-powered rural healthcare support using Google's Gemma model.
</div>

</div>

</div>

</div>

"""


# ============================================================
# BUILD APP
# ============================================================

with gr.Blocks(css=custom_css) as demo:

    gr.HTML(header_html)

    with gr.Row():

        # LEFT SIDE
        with gr.Column(scale=1):

            gr.Markdown("""

# ✨ Why VitaEdge?

### 🧠 AI-powered symptom triage

### 🚨 Urgency classification

### 📋 Explainable reasoning

### 📊 Confidence scoring

### 🏥 Referral recommendations

### 🌍 Offline-first healthcare concept

""")

        # RIGHT SIDE
        with gr.Column(scale=2):

            patient_input = gr.Textbox(
                lines=8,
                label="📋 Patient Information",
                placeholder="Enter patient symptoms and details here..."
            )

            submit_btn = gr.Button("✨ Analyze Patient")

            output_box = gr.Textbox(
                lines=18,
                label="🧠 VitaEdge Analysis"
            )

            submit_btn.click(
                fn=vitaedge_app,
                inputs=patient_input,
                outputs=output_box
            )

    # ========================================================
    # EXAMPLES
    # ========================================================

    gr.Markdown("## 📚 Example Patient Cases")

    gr.Examples(
        examples=examples,
        inputs=patient_input
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.Markdown("""

<div style="
margin-top:20px;
padding:18px;
border-radius:16px;
background:linear-gradient(90deg,#ff2e93,#2563eb);
color:white;
font-size:18px;
font-weight:600;
text-align:center;
">

⚠️ VitaEdge is an educational AI healthcare prototype and not a replacement for licensed medical professionals.

</div>

""")

# ============================================================
# LAUNCH
# ============================================================

demo.launch()
