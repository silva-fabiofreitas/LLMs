from langchain_core.prompts import ChatPromptTemplate

def prompt_image():
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Describe the image provided"),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
            ("system", "{format_instructions}")
        ]
)