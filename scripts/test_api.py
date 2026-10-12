import os
from dotenv import load_dotenv
import anthropic


def main():
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")

    if not api_key or api_key == "여기에_Claude_API_키를_입력하세요":
        print("❌ 오류: .env 파일에 유효한 CLAUDE_API_KEY가 없습니다.")
        return

    print(f"✅ API Key Loaded: {api_key[:10]}... (총 {len(api_key)}자)")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("\n🔍 사용 가능한 Claude 모델 목록을 불러오는 중...")

        models_page = client.models.list()
        models = [model.id for model in models_page.data]
        

        if not models:
            print("⚠️ 사용 가능한 모델이 없습니다.")
        else:
            print(f"\n✨ 총 {len(models)}개의 사용 가능한 모델을 찾았습니다:\n")
            for model_id in models:
                print(f" - {model_id}")

    except Exception as e:
        print(f"\n❌ API 통신 중 오류가 발생했습니다: {e}")
        print("Claude에서 키를 제대로 발급받았는지 확인해주세요.")


if __name__ == "__main__":
    main()
