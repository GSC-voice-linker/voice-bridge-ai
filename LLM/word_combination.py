from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

project_id = "striped-strata-411107"
location = "asia-northeast3"
temperature = 0.9

def mk_sentence(
    temperature: float,
    project_id: str,
    location: str,
    words: list  # Initialize the words list
) -> str:
    """Ideation example with a Large Language Model"""

    vertexai.init(project=project_id, location=location)
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 2048,  # Token limit determines the maximum amount of text output.
        "top_p": 1,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 0,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    # Combine the fixed prompt with the words list
    prompt = "다음의 단어들을 순서대로 이용해서해서 자연스러운 대화형 문장 하나로 만들어줘. 대화형이라고 해서 상대의 대답까지 만들어달라는건 아니야. 예를 들어 주어진 단어가(나 오늘 금요일 기분 좋다)라면 (나는 오늘이 금요일이라 기분이 좋아)라고 자연스럽게 만들어줘. 이때 마지막이 물음표면 의문문으로 만들어줘. " + ",".join(words)
    response = model.predict(
        prompt,
        **parameters,
    )
    print(f"Response from Model: {response.text}")

    return response.text

if __name__ == "__main__":
    mk_sentence(temperature=temperature, project_id=project_id, location=location, words=["word1", "word2", "word3"])
