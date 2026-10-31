# 교수용 평가 데이터 검토표 (v2 파일럿)

## 데이터의 성격

이 30개는 실제 학생 의견 데이터가 아니라 교수용 학생 질문 분석 기능을 시험하기 위한 합성 시나리오입니다. 따라서 빈도·대표성·학생 집단의 실제 요구를 주장하는 근거로 쓸 수 없습니다. 논문 본 실험 전에 교수자 2인 이상이 질문 묶음과 기준 답안을 독립 검토하고, 실제 익명화 질문으로 교체·보강할 것을 권합니다. 각 모델은 동일한 시나리오를 받습니다.

## 시나리오 분포

- `assessment_and_feedback`: 6
- `assignment_and_project_operations`: 6
- `concept_and_prerequisites`: 6
- `learning_environment_and_accessibility`: 6
- `participation_and_communication`: 6

## 문항 목록

| ID | 과목 맥락 | 시나리오 | 문항 수 | 검토 상태 |
|---|---|---|---:|---|
| P001 | 자료구조 | 강의 개념 이해 | 6 | needs_domain_review |
| P002 | 프로그래밍기초 | 과제 요구사항과 제출 | 7 | needs_domain_review |
| P003 | 컴퓨터구조 | 시험 범위와 평가 기준 | 7 | needs_domain_review |
| P004 | 인공지능개론 | 선수지식과 강의 속도 | 6 | needs_domain_review |
| P005 | 캡스톤디자인 | 팀 프로젝트 운영 | 7 | needs_domain_review |
| P006 | 운영체제 | 실습 환경과 설치 문제 | 6 | needs_domain_review |
| P007 | 데이터베이스 | 성적과 피드백 | 6 | needs_domain_review |
| P008 | 컴퓨터네트워크 | 수업 참여와 출석 | 6 | needs_domain_review |
| P009 | 알고리즘 | 학습 자료와 연습 기회 | 6 | needs_domain_review |
| P010 | 소프트웨어공학 | 팀 과제 일정과 소통 | 6 | needs_domain_review |
| P011 | 인공지능개론 | 수업 자료 접근성 | 6 | needs_domain_review |
| P012 | 자료구조 | 상담 및 질문 응답 경로 | 6 | needs_domain_review |
| P013 | 이산수학 | 증명 과정 이해 | 6 | needs_domain_review |
| P014 | 데이터베이스 | 정규화와 스키마 설계 | 6 | needs_domain_review |
| P015 | 컴퓨터네트워크 | IP 주소와 서브넷 | 6 | needs_domain_review |
| P016 | 프로그래밍기초 | 자동 채점과 테스트 케이스 | 6 | needs_domain_review |
| P017 | 소프트웨어공학 | 요구사항 변경과 산출물 | 6 | needs_domain_review |
| P018 | 프로그래밍기초 | 코드 스타일과 제출 점검 | 6 | needs_domain_review |
| P019 | 자료구조 | 부분 점수와 풀이 과정 | 6 | needs_domain_review |
| P020 | 캡스톤디자인 | 발표 평가와 시연 | 6 | needs_domain_review |
| P021 | 컴퓨터구조 | 퀴즈 운영과 피드백 | 6 | needs_domain_review |
| P022 | 알고리즘 | 시험 문제와 수업 목표의 연결 | 6 | needs_domain_review |
| P023 | 운영체제 | 계정과 실습 서버 접근 | 6 | needs_domain_review |
| P024 | 인공지능개론 | 강의자료 접근 형식 | 6 | needs_domain_review |
| P025 | 데이터베이스 | 실습실 장비와 소프트웨어 | 6 | needs_domain_review |
| P026 | 프로그래밍기초 | 실습 중 오류 지원 | 6 | needs_domain_review |
| P027 | 소프트웨어공학 | 팀 내 의견 충돌과 조정 | 6 | needs_domain_review |
| P028 | 인공지능개론 | 질문 채널과 응답 기대 | 6 | needs_domain_review |
| P029 | 자료구조 | 수업 중 질문 참여 | 6 | needs_domain_review |
| P030 | 알고리즘 | 학습 상담과 보충 지원 | 6 | needs_domain_review |

## 교수자 검토 및 승인 절차

각 문항에 대해 두 검토자가 독립적으로 아래 항목을 확인하고, 불일치는 협의해 해결합니다. 승인 전에는 `needs_domain_review` 상태를 유지합니다.

1. 질문이 해당 과목에서 현실적으로 나올 수 있는가?
2. 질문 묶음이 단일한 분석 주제를 갖고 있고, 서로 모순되는 가정을 요구하지 않는가?
3. 기준 요약·키워드·제안이 제공된 질문만으로 도출 가능한가?
4. 권장 제안이 교육적으로 타당하고 과도한 정책·원인 추정을 피하는가?
5. 각 모델 출력에서 동일한 루브릭(주제 포괄성, 근거성, 제안의 구체성, 추론 절제)을 평가할 수 있는가?
6. `reviewer_id`, `reviewed_at`, 수정 이력을 기록했는가?
7. LoRA 학습 자료와 문장·사례 구조가 겹치는지 의미 수준으로 대조했는가? 자동 문자열 중복 확인만으로 누출 없음을 주장하지 않는다.

## 본 실험 전 필수 작업

- 교수자 검토 후 문항을 승인·수정·제외한다.
- 가능하면 익명화한 실제 질문 묶음을 추가하고, 합성/실제 출처를 분리 보고한다.
- 학습 데이터와 평가 데이터를 과목·시나리오 단위로 분리한다. 문장만 바꾼 유사 질문을 평가셋에 남기지 않는다.
- 수정된 30개 문항으로 Base, Student LoRA, Professor LoRA 응답을 모두 새로 생성한다.
- 사람 평가자는 모델명을 가린 결과를 무작위 순서로 채점하고, 최소 2인의 점수와 일치도를 보고한다.
