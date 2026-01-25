Where:
/home/jwang/Agents/lucidgym/rllm/rllm/parser/chat_template_parser.py
What:
elif not self.disable_thinking:
    # generation was cut short during reasoning
    reasoning = completion_text
    if reasoning.startswith("<think>"):
        # og case
        reasoning = reasoning[len("<think>") :]
        reasoning = reasoning.strip()
        content = ""
    else:
        # the model is not a thinking model
        reasoning = ""
        content = completion_text