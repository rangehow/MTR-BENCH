from example import documents,dialog
import random

def generate_prompt(text):

   selected_keys = random.sample(list(documents.keys()), 2)
    
    # 构建两个示例
   def format_documents(key, docs):
        formatted_text = ""
        for i, text in enumerate(docs[key]):
            formatted_text += f"[{key}{i}] {text}\n"
        return formatted_text.strip()
    
    # 获取对应的dialog内容
   documents_1 = format_documents(selected_keys[0], documents)
   documents_2 = format_documents(selected_keys[1], documents)
   dialog_1 = dialog[selected_keys[0]]
   dialog_2 = dialog[selected_keys[1]]
    
   prompt_template = get_prompt_template()
   
    # 5. 填充模板
   filled_prompt = prompt_template.format(
        documents_1=documents_1,
        documents_2=documents_2,
        dialog_1=dialog_1,
        dialog_2=dialog_2,
        text=text 
    )
    
   return filled_prompt, selected_keys

def get_prompt_template():
   return """Your task is to generate document-grounded dialogues where each follow-up question must refer back to content from previous questions through anaphora and ellipsis.

Requirements for the dialogue:

1. **DIALOGUE LENGTH:**

   - Generate 7-10 rounds of dialogue

2. **User questions MUST:**

   a. Focus on one specific paragraph only:
      - The answer to each question must be completely contained within a single paragraph
      - Questions must not:
         * Require combining information from multiple paragraphs
         * Ask for comparisons between different paragraphs or entities (explicitly or implicitly)
         * Need synthesis of information across paragraphs
         * Request summaries of multiple paragraphs
         * Ask for relationships or connections between information in different paragraphs

   b. Questions must use anaphora and ellipsis to refer back to content from previous questions in a natural manner:
      - Anaphoric references to previous questions
         * Using pronouns to refer to entities mentioned in previous questions (it, they, them)
         * Using demonstratives to reference concepts from previous questions (this, that, these, those)
         * Using relative pronouns to connect with prior questions (which, whose)

      - Natural ellipsis in follow-up questions
         * Omitting previously mentioned subjects ("Any success?" instead of "Have these measures had any success?")
         * Shortening questions based on context ("the results?" instead of "What were the results?")
         * Using partial phrases that rely on previous context ("Across all ages?" instead of "Were these findings consistent across all age groups?")

   c. Topic transitions:
      - When source paragraphs cover distinctly different topics:
         * Complete questions without anaphora/ellipsis are acceptable when switching to a new unrelated topic
         * Prioritize natural transitions over forced anaphora/ellipsis
         * Once the new topic is established, resume using anaphora and ellipsis in follow-up questions

3. **Assistant responses MUST:**

   - Be completely independent of the previous answers and must NOT reference or exploit the context of the conversation.
   - Draw information STRICTLY from ONE source paragraph ONLY
   - NEVER combine information from multiple paragraphs in a single response
   - Include exactly one source citation [xxxx] at the end
   - Each response must be traceable to a single specific paragraph
   - Be clear, direct, and informative
   - Extract sufficient details from the source paragraph to support follow-up questions
   - Maintain factual accuracy from the source paragraph

4. **Paragraph usage and citation requirements:**
   - ALL provided paragraphs MUST be used at least once in the dialogue
   - Aim for balanced usage across all paragraphs
   - Citations must accurately match the source of information
   - Multiple references to the same paragraph should focus on different aspects or details

5. **Format requirements:**
   - Follow the exact format shown in the examples
   - Use "User:" and "Assistant:" labels
   - Include paragraph citations in [xxxx] format
   - Maintain consistent formatting throughout

---

### **Valid examples:**

#### Example 1:
Source paragraphs:
{documents_1}

Valid dialogue:
{dialog_1}

#### Example 2:
Source paragraphs:
{documents_2}

Valid dialogue:
{dialog_2}

---

### **Invalid examples (what NOT to do):**

Source paragraphs:
[100] City A, located in the southern region, is a major urban center with a population of 10,000. It has a modern hospital and public library serving local residents.
[102] City B, located 50km north, has 20,000 residents and is known for its excellent schools and shopping centers.

Invalid dialogue examples:
User: Where is City A located?
Assistant: City A is located in the southern region. [100]

User: What's City A's population? (No coreference used - should use "its" instead "City A's")
Assistant: It has 10,000 people. [100]

User: How does that compare to City B? (INVALID - asks to compare paragraphs)
Assistant: City B has 20,000 people, which is twice as many as City A. [100][102] (INVALID - mixes information from multiple paragraphs)

User: Which city has better public services? (INVALID - requires evaluation across multiple paragraphs)
Assistant: City B has better public services due to its excellent schools, while City A focuses more on healthcare. [100][102] (INVALID - evaluates and synthesizes information across paragraphs)

User: Which city is better overall? (INVALID - requires evaluating multiple paragraphs)
Assistant: Based on the facilities and population, both cities have their advantages. [100][102] (INVALID - draws conclusions from multiple paragraphs)

---

### **Conclusion:**
Following the requirements and examples provided above, generate a document-grounded dialogue for the paragraphs below, ensuring proper use of anaphora and ellipsis:

Source paragraphs: 
{text}

Valid dialogue:
"""

