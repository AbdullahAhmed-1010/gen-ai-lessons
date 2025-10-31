from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template='''
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Equations:
    Include relevant mathematical equations if present in the paper.
    Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Use practical analogies to simplify complex ideas.
    Where sufficient information is not available in the paper, respond with “insufficient information”.
    Make sure the explanation is clear, accurate and aligned with the prescribed style and length.
    ''',
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

template.save("template.json")