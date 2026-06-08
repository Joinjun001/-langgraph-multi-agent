# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# Python 바이트코드(.pyc) 생성 방지 및 버퍼링 비활성화
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 컴파일이나 빌드에 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 코드 복사
COPY . .

# FastAPI 컨테이너 포트 개방
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "app.py"]
