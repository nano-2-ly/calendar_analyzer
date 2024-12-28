from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import icalendar
import pytz
import openai
import json
from tqdm import tqdm

# 캘린더 API 범위
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def calculate_week_percentage(service, week_start, week_end):
    time_min = week_start.astimezone(timezone.utc).isoformat()
    time_max = week_end.astimezone(timezone.utc).isoformat()
    
    events_result = service.events().list(calendarId='ding991207@gmail.com',
                                        timeMin=time_min,
                                        timeMax=time_max,
                                        maxResults=1000,
                                        singleEvents=True,
                                        orderBy='startTime').execute()
    events = events_result.get('items', [])
    
    time_intervals = []
    for event in events:
        start_str = event['start'].get('dateTime', event['start'].get('date'))
        end_str = event['end'].get('dateTime', event['end'].get('date'))
        
        try:
            # 시간대 정보가 없는 경우 UTC로 가정
            if 'T' in start_str:  # datetime 형식인 경우
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
            else:  # date 형식인 경우 건너뛰기
                continue
            
            time_intervals.append((start_time, end_time))
        except ValueError:
            continue
    
    # 시간 간격 정렬 및 병합
    time_intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    if time_intervals:
        merged_intervals = [time_intervals[0]]
        for current in time_intervals[1:]:
            previous = merged_intervals[-1]
            if current[0] <= previous[1]:
                merged_intervals[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged_intervals.append(current)
    
    total_scheduled_minutes = sum(
        (interval[1] - interval[0]).total_seconds() / 60
        for interval in merged_intervals
    )
    
    total_week_minutes = 7 * 24 * 60
    return (total_scheduled_minutes / total_week_minutes) * 100

def calculate_day_percentage(service, day_start, day_end):
    time_min = day_start.astimezone(timezone.utc).isoformat()
    time_max = day_end.astimezone(timezone.utc).isoformat()
    
    events_result = service.events().list(calendarId='ding991207@gmail.com',
                                        timeMin=time_min,
                                        timeMax=time_max,
                                        maxResults=1000,
                                        singleEvents=True,
                                        orderBy='startTime').execute()
    events = events_result.get('items', [])
    
    time_intervals = []
    for event in events:
        start_str = event['start'].get('dateTime', event['start'].get('date'))
        end_str = event['end'].get('dateTime', event['end'].get('date'))
        
        try:
            if 'T' in start_str:
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                continue
            
            time_intervals.append((start_time, end_time))
        except ValueError:
            continue
    
    time_intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    if time_intervals:
        merged_intervals = [time_intervals[0]]
        for current in time_intervals[1:]:
            previous = merged_intervals[-1]
            if current[0] <= previous[1]:
                merged_intervals[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged_intervals.append(current)
    
    total_scheduled_minutes = sum(
        (interval[1] - interval[0]).total_seconds() / 60
        for interval in merged_intervals
    )
    
    total_day_minutes = 24 * 60
    return (total_scheduled_minutes / total_day_minutes) * 100

class CalendarAnalyzer:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.categories = [
            "업무", "휴식", "수면", "식사", "운동", "기타", "오류"
        ]
    
    def classify_event(self, event_summary, event_description=""):
        # LLM에게 보낼 프롬프 작성
        prompt = f"""
다음 일정이 어느 카테고리에 속하는지 분류해주세요.
카테고리: {', '.join(self.categories)}

일정 제목: {event_summary}
일정 설명: {event_description}

답변은 카테고리 중 딱 한 단어로만 작성해주세요. 카테고리 중 없는 경우는 오류로 작성해주세요.

"""
        
        # LLM API 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 일정을 정확하게 분류하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()

    def analyze_calendar(self, ics_file_path):
        # 결과를 저장할 딕셔너리
        time_spent = {cat: 0 for cat in self.categories}
        events_by_category = {cat: [] for cat in self.categories}
        
        # .ics 파일 읽기
        with open(ics_file_path, 'rb') as file:
            cal = icalendar.Calendar.from_ical(file.read())
            
        # 각 이벤트 분석
        for event in cal.walk('vevent'):
            start = event.get('dtstart').dt
            end = event.get('dtend').dt
            summary = str(event.get('summary', ''))
            description = str(event.get('description', ''))
            
            # LLM을 통한 카테고리 분류
            category = self.classify_event(summary, description)
            
            # 시간 계산 (분 단위)
            duration = (end - start).total_seconds() / 60
            time_spent[category] += duration
            
            # 이벤트 정보 저장
            events_by_category[category].append({
                'summary': summary,
                'start': start,
                'end': end,
                'duration': duration
            })
        
        # 분을 시간으로 변환
        time_spent = {k: round(v/60, 1) for k, v in time_spent.items()}
        
        return time_spent, events_by_category

    def analyze_and_save(self, service):
        # 1년치 데이터 가져오기
        now = datetime.now(timezone.utc)
        year_start = now - timedelta(days=365)
        
        time_min = year_start.isoformat()
        time_max = now.isoformat()
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=2500,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        categorized_events = {cat: [] for cat in self.categories}
        
        # tqdm을 사용하여 프로그레스 바 추가
        for event in tqdm(events, desc="일정 분석 중"):
            summary = event.get('summary', '')
            description = event.get('description', '')
            
            # 이벤트 시간 처리
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            if 'T' not in start:  # 종일 일정은 건너뛰기
                continue
                
            # 카테고리 분류
            category = self.classify_event(summary, description)
            
            # 시간 계산
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
            duration = (end_time - start_time).total_seconds() / 3600  # 시간 단위
            
            try:
                categorized_events[category].append({
                    'summary': summary,
                    'start': start,
                    'end': end,
                    'duration': duration
                })
            except KeyError:
                category = '오류'
                categorized_events[category].append({
                    'summary': summary,
                    'start': start,
                    'end': end,
                    'duration': duration
                })
        
        # 결과 저장
        with open('calendar_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(categorized_events, f, ensure_ascii=False, indent=2)
        
        return categorized_events

def visualize_categories(categorized_events):
    # 카테고리별 총 시간 계산
    category_hours = {}
    for category, events in categorized_events.items():
        total_hours = sum(event['duration'] for event in events)
        category_hours[category] = total_hours
    
    # 그래프 생성
    plt.figure(figsize=(12, 6))
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    # 막대 그래프 생성
    categories = list(category_hours.keys())
    hours = list(category_hours.values())
    
    plt.bar(categories, hours)
    plt.title('연간 카테고리별 소요 시간', pad=20)
    plt.xlabel('카테고리')
    plt.ylabel('총 소요 시간 (시간)')
    
    # 막대 위에 시간 표시
    for i, v in enumerate(hours):
        plt.text(i, v, f'{v:.1f}h', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('category_analysis.png')
    
    # 통계 출력
    print("\n[카테고리별 연간 소요 시간]")
    for category, hours in category_hours.items():
        print(f"{category}: {hours:.1f}시간")

def analyze_saved_data():
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지
    
    # JSON 파일 로드
    with open('calendar_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터프레임으로 변환하기 위한 리스트 생성
    events_list = []
    for category, events in data.items():
        for event in events:
            start_time = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
            events_list.append({
                'category': category,
                'start_time': start_time,
                'duration': event['duration'],
                'month': start_time.month,
                'week': start_time.isocalendar()[1],
                'weekday': start_time.weekday()
            })
    
    df = pd.DataFrame(events_list)
    
    # 1. 카테고리별 연간 총 사용시간
    plt.figure(figsize=(12, 6))
    annual_usage = df.groupby('category')['duration'].sum()
    annual_usage.plot(kind='bar')
    plt.title('카테고리별 연간 총 사용시간')
    plt.xlabel('카테고리')
    plt.ylabel('시간')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('annual_category_usage.png')
    plt.close()
    
    # 2. 월간 카테고리별 총 사용시간
    plt.figure(figsize=(15, 8))
    monthly_usage = df.pivot_table(
        values='duration',
        index='month',
        columns='category',
        aggfunc='sum'
    ).fillna(0)
    monthly_usage.plot(kind='bar', stacked=True)
    plt.title('월간 카테고리별 총 사용시간')
    plt.xlabel('월')
    plt.ylabel('시간')
    plt.legend(title='카테고리', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('monthly_category_usage.png')
    plt.close()
    
    # 3. 주간 카테고리별 총 사용시간
    plt.figure(figsize=(20, 8))
    weekly_usage = df.pivot_table(
        values='duration',
        index='week',
        columns='category',
        aggfunc='sum'
    ).fillna(0)
    weekly_usage.plot(kind='bar', stacked=True)
    plt.title('주간 카테고리별 총 사용시간')
    plt.xlabel('주차')
    plt.ylabel('시간')
    plt.legend(title='카테고리', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('weekly_category_usage.png')
    plt.close()
    
    # 4. 요일별 카테고리별 평균 사용시간
    plt.figure(figsize=(12, 6))
    weekday_avg = df.pivot_table(
        values='duration',
        index='weekday',
        columns='category',
        aggfunc='mean'
    ).fillna(0)
    weekday_avg.index = ['월', '화', '수', '목', '금', '토', '일']
    weekday_avg.plot(kind='bar', stacked=True)
    plt.title('��일별 카테고리별 평균 사용시간')
    plt.xlabel('요일')
    plt.ylabel('시간')
    plt.legend(title='카테고리', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('weekday_category_avg.png')
    plt.close()
    
    # 통계 출력
    print("\n[카테고리별 연간 총 사용시간]")
    print(annual_usage)
    
    print("\n[월간 평균 사용시간]")
    print(monthly_usage.mean())
    
    print("\n[요일별 평균 사용시간]")
    print(weekday_avg)

def main():
    creds = None
    # 이전에 저장된 토큰이 있으면 로드
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # 인증이 필요하면 새로 수행
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        # 인증 정보를 저장
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Google Calendar API 빌드
    service = build('calendar', 'v3', credentials=creds)
    
    # 현재 날짜 (timezone 정보 포함)
    now = datetime.now(timezone.utc)
    
    # 데이터 수집
    weeks_data = []
    week_labels = []
    
    # for i in range(50):
    #     week_end = now - timedelta(days=now.weekday(), weeks=i)
    #     week_start = week_end - timedelta(days=7)
    #     percentage = calculate_week_percentage(service, week_start, week_end)
    #     weeks_data.append(percentage)
    #     # 한국 시간으로 변환하여 레이블 생성
    #     kr_time = week_start.astimezone(timezone(timedelta(hours=9)))
    #     week_labels.append(f"{kr_time.strftime('%m/%d')}")
    
    # # 데이터 역순으로 변경 (과거->현재)
    # weeks_data.reverse()
    # week_labels.reverse()
    
    # # 그래프 그리기
    # plt.figure(figsize=(15, 6))
    
    # # 한글 폰트 설정 (맑은 고딕 사용)
    # plt.rcParams['font.family'] = 'Malgun Gothic'
    
    # plt.plot(weeks_data, marker='o')
    # plt.title('주간 일정 비율 추이 (최근 50주)', pad=20)
    # plt.xlabel('날짜')
    # plt.ylabel('주간 일정 비율 (%)')
    
    # # x축 레이블 설정
    # plt.xticks(range(0, 50, 5), [week_labels[i] for i in range(0, 50, 5)], rotation=45)
    
    # # 그리드 추가
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # # 여백 조정
    # plt.tight_layout()
    
    # # 그래프 저장
    # plt.savefig('weekly_schedule_trend.png')
    
    # # 통계 출력
    # print("\n[주간 일정 통계]")
    # print(f"평균 일정 비율: {sum(weeks_data)/len(weeks_data):.1f}%")
    # print(f"최대 일정 비율: {max(weeks_data):.1f}%")
    # print(f"최소 일정 비율: {min(weeks_data):.1f}%")

    # # 요일별 데이터 수집
    # days_data = {i: [] for i in range(7)}  # 0=월요일, 6=일요일
    # now = datetime.now(timezone.utc)
    
    # # 최근 52주(1년) 동안의 데이터 수집
    # for week in range(52):
    #     week_end = now - timedelta(days=now.weekday(), weeks=week)
    #     for day in range(7):
    #         day_start = week_end - timedelta(days=7-day)
    #         day_end = day_start + timedelta(days=1)
    #         percentage = calculate_day_percentage(service, day_start, day_end)
    #         days_data[day].append(percentage)
    
    # # 요일별 평균 계산
    # day_averages = {day: sum(percentages)/len(percentages) for day, percentages in days_data.items()}
    
    # # 두 번째 그래프: 요일별 평균 일정 비율
    # plt.figure(figsize=(10, 6))
    # days_kr = ['월', '화', '수', '목', '금', '토', '일']
    
    # plt.bar(days_kr, [day_averages[i] for i in range(7)])
    # plt.title('요일별 평균 일정 비율 (최근 1년)', pad=20)
    # plt.xlabel('요일')
    # plt.ylabel('평균 일정 비율 (%)')
    
    # # 그리드 추가
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # # 여백 조정
    # plt.tight_layout()
    
    # # 그래프 저장
    # plt.savefig('daily_schedule_trend.png')
    
    # # 요일별 통계 출력
    # print("\n[요일별 평균 일정 비율]")
    # for day in range(7):
    #     print(f"{days_kr[day]}요일: {day_averages[day]:.1f}%")

    # # 캘린더 분석
    # api_key = os.getenv('OPENAI_API_KEY')  # OpenAI API 키를 .env 파일에서 불러오기
    # analyzer = CalendarAnalyzer(api_key)
    # categorized_events = analyzer.analyze_and_save(service)
    
    # # 시각화
    # visualize_categories(categorized_events)

    # # 그래프 생성 추가
    # analyze_saved_data()

if __name__ == '__main__':
    main()
