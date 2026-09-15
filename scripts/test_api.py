import os

import google.generativeai as genai
from dotenv import load_dotenv


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "여기에_Gemini_API_키를_입력하세요":
        print("❌ 오류: .env 파일에 유효한 GOOGLE_API_KEY가 없습니다.")
        return

    print(f"✅ API Key Loaded: {api_key[:10]}... (총 {len(api_key)}자)")

    try:
        genai.configure(api_key=api_key)
        print("\n🔍 사용 가능한 Gemini 모델 목록을 불러오는 중...")

        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append(m.name)
                print(f" - {m.name}")

        if not models:
            print("⚠️ 사용 가능한 모델이 없습니다. API 키 권한이나 구글 클라우드 설정을 확인하세요.")
        else:
            print(f"\n✨ 총 {len(models)}개의 모델이 사용 가능합니다.")

    except Exception as e:
        print(f"\n❌ API 통신 중 오류가 발생했습니다: {e}")
        print("Google AI Studio에서 키를 제대로 발급받았는지 확인해주세요.")


if __name__ == "__main__":
    main()
