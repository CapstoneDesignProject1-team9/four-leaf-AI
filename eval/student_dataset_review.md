# 학생용 평가 질문 검토표 (파일럿 v2)

범위: RAG 지식베이스가 포함하는 2026학년도 2학기 자료. 같은 ID의 문항을 Base / Base+RAG / Student LoRA / Professor LoRA / Student LoRA+RAG 조건에 모두 사용합니다.
현재 이 문서는 문항 구성표입니다. 공식 출처의 URL·제목·게시일 확인 및 교수자/학사 담당자 검토가 끝나기 전에는 승인된 정답 집합으로 간주하지 않습니다.

| ID | 주제 | 질문 | 기준/검토 상태 |
|---|---|---|---|
| S001 | curriculum_and_major | 인공지능컴퓨팅전공(인컴)의 전공필수 과목이 뭐가 있어? | needs_official_source_verification |
| S002 | curriculum_and_major | 인컴 전공에서 타 학과 개설 동일 교과목도 전공으로 인정돼? | needs_official_source_verification |
| S003 | graduation_and_student_status | ABEEK 이수 포기는 어떻게 신청해? | needs_official_source_verification |
| S004 | curriculum_and_major | 플솝에서 ABEEK 인정되는 과목 기준이 어떻게 돼? | needs_official_source_verification |
| S005 | curriculum_and_major | 플솝 2023학번 이상 학생의 TOPCIT 관련 졸업요건이 뭐야? | needs_official_source_verification |
| S006 | out_of_scope_abstention | 2027학년도 1학기 수강변경 기간은 언제야? | designed_out_of_scope |
| S007 | registration_and_academic_calendar | 수강정정 기간에는 뭘 할 수 있어? | needs_official_source_verification |
| S008 | registration_and_academic_calendar | 2026학년도 2학기 휴학 신청 기간 알려줘 | needs_official_source_verification |
| S009 | registration_and_academic_calendar | 복학은 언제까지 신청해야 해? 군 제대자 주의사항도 알려줘 | needs_official_source_verification |
| S010 | registration_and_academic_calendar | 2026학년도 2학기 등록금 납부 일정이 어떻게 돼? | needs_official_source_verification |
| S011 | student_financial_or_travel_support | 국가장학금 2차 신청은 언제 어떻게 해? | needs_official_source_verification |
| S012 | student_financial_or_travel_support | 주거안정장학금은 누가 신청할 수 있어? | needs_official_source_verification |
| S013 | graduation_and_student_status | 컴퓨터학부 지도교수 확인하고 변경하려면 어떻게 해? | needs_official_source_verification |
| S014 | graduation_and_student_status | 경북대 졸업자격인정제가 뭐야? 어떤 요건을 충족해야 해? | needs_official_source_verification |
| S015 | graduation_and_student_status | 글솝 졸업예정자는 어떤 준비를 해야 해? | needs_official_source_verification |
| S016 | graduation_and_student_status | 졸업증서는 언제부터 받을 수 있어? | needs_official_source_verification |
| S017 | graduation_and_student_status | 플솝 졸업예정자 졸업요건확인서는 언제까지 제출해야 해? | needs_official_source_verification |
| S018 | curriculum_and_major | 자료구조 과목 수강할 때 주의할 점이 있어? | needs_official_source_verification |
| S019 | registration_and_academic_calendar | 2026학번 신입생 수강꾸러미 신청은 어떻게 해? | needs_official_source_verification |
| S020 | student_financial_or_travel_support | 종합설계 프로젝트 재료비 지원은 어떻게 신청해? | needs_official_source_verification |
| S021 | student_financial_or_travel_support | 종합설계 여비 지원은 어떻게 받아? | needs_official_source_verification |
| S022 | course_and_student_operations | 조기취업자 출석인정 신청하려면 어떻게 해야 돼? | needs_official_source_verification |
| S023 | course_and_student_operations | 캡스톤디자인 운영에 참여하려면 어떤 조건이 필요해? | needs_official_source_verification |
| S024 | curriculum_and_major | 2026학년도 2학기 융합전공 신청 조건이랑 일정 알려줘 | needs_official_source_verification |
| S025 | course_and_student_operations | 학칙 개정으로 수강신청 제도가 어떻게 바뀌어? | needs_official_source_verification |
| S026 | course_and_student_operations | 성균관대학교 학점교류 신청 어떻게 해? | needs_official_source_verification |
| S027 | course_and_student_operations | 2026학년도 2학기 안전교육은 어떻게 이수해? | needs_official_source_verification |
| S028 | registration_and_academic_calendar | 여름계절수업 성적 열람이랑 이의 신청은 언제 할 수 있어? | needs_official_source_verification |
| S029 | out_of_scope_abstention | 2027학년도 1학기 융합전공 신청 기간과 선발 인원은 어떻게 돼? | designed_out_of_scope |
| S030 | course_and_student_operations | 학부사무실 문의 이메일 주소가 뭐야? | needs_official_source_verification |

## 분포

- `curriculum_and_major`: 6문항
- `registration_and_academic_calendar`: 6문항
- `graduation_and_student_status`: 6문항
- `student_financial_or_travel_support`: 4문항
- `course_and_student_operations`: 6문항
- `out_of_scope_abstention`: 2문항

## 검토자가 확인할 항목

1. factual_answer 문항마다 공식 공지/규정 URL, 제목, 게시일(또는 적용일), 확인일을 기록한다.
2. 정답을 원자적 필수 사실로 분해하고 적용 학기·학번·전공·대상을 검증한다.
3. 학생 질문의 자연스러움과 답변 가능성을 확인하고, 출처가 없거나 기준이 모호한 문항은 수정 또는 제외한다.
4. S006·S029는 2026학년도 2학기 지식베이스 범위를 벗어나는 abstention 평가 문항이다. 모델이 구체적인 날짜/인원을 추측하면 오답으로 판정한다.
5. 본 실험 전 모든 답변과 검색 문맥을 새 문항 텍스트 기준으로 다시 생성한다. 예전 모델 출력은 비교에 사용하지 않는다.
6. 정답·문항 수정 이력, 검토자 코드, 검토일을 남긴다. 실제 학생 질의를 사용할 경우 개인 식별정보를 제거한다.
