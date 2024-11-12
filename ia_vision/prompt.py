from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    
def prompt_images():
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Analyze the images and classify them into categories, then synthesize the result into a Pareto analysis."),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ]
)