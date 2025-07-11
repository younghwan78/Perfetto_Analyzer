import pandas as pd
import json
from typing import Dict
import re

class SystraceEngine:
    """
    엔진: systrace 파일 파싱 및 event/interrupt/latency 분석 기능 제공
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.df = None

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Loaded configuration for {len(config['events'])} events")
                return config
        except FileNotFoundError:
            print(f"Error: Config file {config_path} not found.")
            return {"events": []}

    def parse_systrace(self, trace_path: str, max_lines: int = None) -> None:
        events = []
        print(f"Parsing systrace file: {trace_path}")
        try:
            with open(trace_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_lines is not None and line_num > max_lines:
                        print(f"Stopped reading at line {max_lines}")
                        break
                    try:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        timestamp_idx = next(i for i, part in enumerate(parts) if '....' in part)
                        timestamp = float(parts[timestamp_idx + 1].rstrip(':'))
                        event_type = parts[timestamp_idx + 2].rstrip(':')
                        process = parts[0]
                        cpu = parts[3].strip('[]')
                        details = ' '.join(parts[timestamp_idx + 3:])
                        events.append({
                            'timestamp': timestamp,
                            'event_type': event_type,
                            'process': process,
                            'cpu': cpu,
                            'details': details
                        })
                        if line_num % 100000 == 0:
                            print(f"Processed {line_num} lines...")
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {str(e)}")
                        continue
            if not events:
                print("Warning: No valid events found in the trace file.")
                print("Please check if the file format is correct.")
                print("Example expected format:")
                print("<...>-1234 [001] .... 123456.789: sched_switch: details...")
                return
            self.df = pd.DataFrame(events)
            print(f"Successfully parsed {len(self.df)} events")
            event_types = self.df['event_type'].unique()
            print("\nFound event types:")
            for evt_type in event_types:
                count = len(self.df[self.df['event_type'] == evt_type])
                print(f"  - {evt_type}: {count} events")
        except FileNotFoundError:
            print(f"Error: Trace file not found: {trace_path}")
            raise
        except Exception as e:
            print(f"Error parsing trace file: {str(e)}")
            raise

    def analyze_event(self, event_config: Dict) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            print("No events to analyze. Please check if the trace file was parsed correctly.")
            return pd.DataFrame()
        event_name = event_config['event_name']
        start_condition = event_config['start_condition']
        end_condition = event_config['end_condition']
        wake_condition = event_config.get('wake_condition')
        # start_condition, end_condition이 리스트인지 확인
        start_conditions = event_config['start_condition']
        if not isinstance(start_conditions, list):
            start_conditions = [start_conditions]
        end_conditions = event_config['end_condition']
        if not isinstance(end_conditions, list):
            end_conditions = [end_conditions]
        intervals = []
        wakeup_time = None
        waiting_time = None
        start_time = None
        start_cpu = None
        start_process = None
        for idx, row in self.df.iterrows():
            # wakeup condition (기존대로 단일 조건)
            if wake_condition and row['event_type'] == wake_condition['event']:
                wake_pattern = f"{wake_condition['match_field']}={wake_condition['match_value']}"
                if wake_pattern in row['details']:
                    wakeup_time = row['timestamp']
            # start condition (OR)
            for cond in start_conditions:
                start_pattern = f"{cond['match_field']}={cond['match_value']}"
                if row['event_type'] == cond['event'] and start_pattern in row['details']:
                    start_time = row['timestamp']
                    start_cpu = row['cpu']
                    start_process = row['process']
                    if wakeup_time is not None:
                        waiting_time = start_time - wakeup_time
                    else:
                        waiting_time = None
                    break  # 여러 조건 중 하나만 만족하면 됨
            # end condition (OR)
            for cond in end_conditions:
                end_pattern = f"{cond['match_field']}={cond['match_value']}"
                if row['event_type'] == cond['event'] and end_pattern in row['details']:
                    if start_time is not None:
                        runtime = row['timestamp'] - start_time
                        intervals.append({
                            'event_name': event_name,
                            'event_type': event_config['event_type'],
                            'wakeup_time': wakeup_time if wakeup_time is not None else None,
                            'start_time': start_time,
                            'end_time': row['timestamp'],
                            'runtime': runtime,
                            'waiting_time': waiting_time if waiting_time is not None else None,
                            'cpu': start_cpu,
                            'start_process': start_process,
                            'end_process': row['process']
                        })
                    start_time = None
                    start_cpu = None
                    start_process = None
                    waiting_time = None
                    wakeup_time = None
                    break  # 여러 조건 중 하나만 만족하면 됨
        result_df = pd.DataFrame(intervals)
        if len(result_df) == 0:
            print(f"Warning: No matching event pairs found for {event_name}")
        
        # --- Merge intervals if gap <= merge_gap_msec ---
        merge_gap_msec = event_config.get('merge_gap_msec')
        if merge_gap_msec is not None and len(result_df) > 1:
            merged = []
            result_df = result_df.sort_values('start_time').reset_index(drop=True)
            cur = result_df.iloc[0].copy()
            for i in range(1, len(result_df)):
                next_row = result_df.iloc[i]
                gap = next_row['start_time'] - cur['end_time']
                if gap * 1000 <= merge_gap_msec:
                    # merge: end_time, runtime만 합침, waiting_time은 첫 interval의 값만 유지
                    cur['end_time'] = next_row['end_time']
                    cur['runtime'] += next_row['runtime']
                    # waiting_time은 합치지 않고 cur의 값을 유지
                else:
                    merged.append(cur)
                    cur = next_row.copy()
            merged.append(cur)
            result_df = pd.DataFrame(merged)
        
        return result_df

    def analyze_latency(self, event_config: Dict) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            print("No events to analyze.")
            return pd.DataFrame()
        event_name = event_config['event_name']
        start_condition = event_config['start_event_condition']
        end_condition = event_config['end_event_condition']
        mid_condition = event_config.get('mid_event_condition')
        
        start_df = self.df[self.df['event_type'] == start_condition['event']].copy()
        end_df = self.df[self.df['event_type'] == end_condition['event']].copy()
        
        if len(start_df) == 0 or len(end_df) == 0:
            print(f"Warning: Not enough events to calculate latency for {event_name}")
            return pd.DataFrame()
        
        start_pattern = f"{start_condition['match_field']}={start_condition['match_value']}"
        start_events = start_df[start_df['details'].str.contains(start_pattern, na=False)].copy()
        start_events['latency_event_type'] = 'start'
        
        end_pattern = f"{end_condition['match_field']}={end_condition['match_value']}"
        end_events = end_df[end_df['details'].str.contains(end_pattern, na=False)].copy()
        end_events['latency_event_type'] = 'end'
        
        # 시간 순서대로 정렬
        start_events = start_events.sort_values('timestamp').reset_index(drop=True)
        end_events = end_events.sort_values('timestamp').reset_index(drop=True)
        
        latencies = []
        # mid_event_condition이 있는 경우
        if mid_condition:
            mid_df = self.df[self.df['event_type'] == mid_condition['event']].copy()
            mid_pattern = f"{mid_condition['match_field']}={mid_condition['match_value']}"
            mid_df = mid_df[mid_df['details'].str.contains(mid_pattern, na=False)].copy()
            mid_df = mid_df.sort_values('timestamp').reset_index(drop=True)
            if len(mid_df) == 0:
                print(f"Warning: No matching mid events found for latency '{event_name}'")
                return pd.DataFrame()
            for _, mid in mid_df.iterrows():
                # mid 이전의 마지막 start
                prev_starts = start_events[start_events['timestamp'] <= mid['timestamp']]
                if prev_starts.empty:
                    continue
                last_start = prev_starts.iloc[-1]
                # mid 이후의 첫 end
                next_ends = end_events[end_events['timestamp'] >= mid['timestamp']]
                if next_ends.empty:
                    continue
                first_end = next_ends.iloc[0]
                # latency 계산 (동시 발생시 0)
                latency = max(0, first_end['timestamp'] - last_start['timestamp'])
                latencies.append({
                    'event_name': event_name,
                    'event_type': event_config['event_type'],
                    'start_time': last_start['timestamp'],
                    'mid_time': mid['timestamp'],
                    'end_time': first_end['timestamp'],
                    'latency': latency,
                    'cpu': last_start['cpu'],
                    'start_process': last_start['process'],
                    'end_process': first_end['process']
                })
        else:
            # mid가 없으면 start~start 구간마다 첫 end만 latency로 사용
            start_times = start_events['timestamp'].tolist()
            end_times = end_events['timestamp'].tolist()
            end_idx = 0
            for i, start_time in enumerate(start_times):
                # 다음 start까지의 구간
                next_start_time = start_times[i+1] if i+1 < len(start_times) else float('inf')
                # 이 구간 내에서 첫 end 찾기
                while end_idx < len(end_times) and end_times[end_idx] < start_time:
                    end_idx += 1  # start 이전 end는 스킵
                if end_idx < len(end_times) and start_time <= end_times[end_idx] < next_start_time:
                    # latency 계산 (동시 발생시 0)
                    latency = max(0, end_times[end_idx] - start_time)
                    last_start = start_events.iloc[i]
                    end_row = end_events[end_events['timestamp'] == end_times[end_idx]].iloc[0]
                    latencies.append({
                        'event_name': event_name,
                        'event_type': event_config['event_type'],
                        'start_time': start_time,
                        'end_time': end_times[end_idx],
                        'latency': latency,
                        'cpu': last_start['cpu'],
                        'start_process': last_start['process'],
                        'end_process': end_row['process']
                    })
                    end_idx += 1  # 다음 start 전까지의 end는 무시
        result_df = pd.DataFrame(latencies)
        if len(result_df) == 0:
            print(f"Warning: No matching latency event pairs found for {event_name}")
        else:
            print(f"Successfully calculated {len(result_df)} latency measurements for {event_name}")
        return result_df

    def analyze_interrupt(self, event_config: Dict) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            print("No events to analyze. Please check if the trace file was parsed correctly.")
            return pd.DataFrame()
        event_name = event_config['event_name']
        start_condition = event_config['start_condition']
        id_field = event_config['id_field']
        id_value_config = event_config.get('id_value')
        id_regex = event_config.get('id_regex')
        end_event = event_config['end_event']
        end_contains = event_config['end_contains']
        start_df = self.df[self.df['event_type'] == start_condition['event']].copy()
        start_pattern = f"{start_condition['match_field']}={start_condition['match_value']}"
        start_events = start_df[start_df['details'].str.contains(start_pattern, na=False)].copy()
        if len(start_events) == 0:
            print(f"Warning: No start events found matching pattern '{start_pattern}'")
            return pd.DataFrame()
        intervals = []
        import re
        for _, start in start_events.iterrows():
            if id_value_config is not None:
                id_value = str(id_value_config)
            else:
                regex_pattern = id_regex if id_regex else rf"{id_field}=(\\d+)"
                id_match = re.search(regex_pattern, start['details'])
                if not id_match:
                    continue
                id_value = id_match.group(1)
            end_df = self.df[(self.df['event_type'] == end_event) &
                             (self.df['details'].str.contains(f"{id_field}={id_value}", na=False)) &
                             (self.df['details'].str.contains(end_contains, na=False)) &
                             (self.df['timestamp'] > start['timestamp'])]
            if not end_df.empty:
                end = end_df.iloc[0]
                runtime = end['timestamp'] - start['timestamp']
                intervals.append({
                    'event_name': event_name,
                    'event_type': event_config['event_type'],
                    'start_time': start['timestamp'],
                    'end_time': end['timestamp'],
                    'runtime': runtime,
                    'cpu': start['cpu'],
                    'start_process': start['process'],
                    'end_process': end['process'],
                    id_field: id_value
                })
        result_df = pd.DataFrame(intervals)
        if len(result_df) == 0:
            print(f"Warning: No matching interrupt event pairs found for {event_name}")
        interval_tagging = event_config.get('interval_tagging')
        if interval_tagging and len(result_df) > 1:
            interval_msec = float(interval_tagging.get('interval_msec', 0))
            fs_tag = interval_tagging.get('fs_tag', 'FS')
            fe_tag = interval_tagging.get('fe_tag', 'FE')
            tags = [''] * len(result_df)
            for i in range(len(result_df) - 1):
                cur_end = result_df.iloc[i]['end_time']
                next_start = result_df.iloc[i+1]['start_time']
                gap_msec = (next_start - cur_end) * 1000
                # msec 단위로만 비교하고 소숫점 단위 변화 허용
                if abs(gap_msec - interval_msec) < 1.0:  # 1ms 이내의 차이는 허용
                    tags[i] = fs_tag
                    tags[i+1] = fe_tag
            result_df['tag'] = tags
        else:
            result_df['tag'] = ''
        # --- Frame Numbering Logic ---
        frame_numbering_config = event_config.get('frame_numbering_config')
        if frame_numbering_config:
            frame_number_event_name = frame_numbering_config.get('frame_number_event_name', 'FS')
            start_frame_number = frame_numbering_config.get('start_frame_number', 1)
            frame_numbers = []
            current_frame = start_frame_number - 1
            for tag in result_df['tag']:
                if tag == frame_number_event_name:
                    current_frame += 1
                frame_numbers.append(current_frame if current_frame >= start_frame_number else None)
            result_df['frame_number'] = frame_numbers
        return result_df

    def analyze_complex_latency(self, event_config: dict, interrupt_results: dict) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            print("No events to analyze.")
            return pd.DataFrame()
        start_interrupt_cfg = event_config['start_interrupt']
        end_interrupt_cfg = event_config['end_interrupt']
        mid_event_cfgs = event_config['mid_event']
        # mid_event가 dict면 리스트로 변환
        if isinstance(mid_event_cfgs, dict):
            mid_event_cfgs = [mid_event_cfgs]
        start_event_name = start_interrupt_cfg['event_name']
        if start_event_name not in interrupt_results:
            print(f"No interrupt results for start event_name: {start_event_name}")
            return pd.DataFrame()
        start_df = interrupt_results[start_event_name]
        if 'tag' in start_interrupt_cfg:
            start_df = start_df[start_df['tag'] == start_interrupt_cfg['tag']]
        if start_df.empty:
            print(f"No start interrupt events matching {start_interrupt_cfg}.")
            return pd.DataFrame()
        end_event_name = end_interrupt_cfg['event_name']
        if end_event_name not in interrupt_results:
            print(f"No interrupt results for end event_name: {end_event_name}")
            return pd.DataFrame()
        end_df = interrupt_results[end_event_name]
        if 'tag' in end_interrupt_cfg:
            end_df = end_df[end_df['tag'] == end_interrupt_cfg['tag']]
        if end_df.empty:
            print(f"No end interrupt events matching {end_interrupt_cfg}.")
            return pd.DataFrame()
        # 모든 mid_event 후보를 모아둠
        mid_event_dfs = []
        for mid_cfg in mid_event_cfgs:
            mid_df = self.df[self.df['event_type'] == mid_cfg['event_type']]
            mid_pattern = f"{mid_cfg['match_field']}={mid_cfg['match_value']}"
            mid_df = mid_df[mid_df['details'].str.contains(mid_pattern, na=False)].copy()
            if mid_df.empty:
                print(f"No mid events matching {mid_pattern}.")
                return pd.DataFrame()
            mid_event_dfs.append(mid_df)
        # 결과 저장
        results = []
        debug_info = []  # 디버깅 정보 저장
        
        # start~end 구간을 순회하며, 각 구간 내에 mid_event들이 순서대로 모두 등장하는지 체크
        for start_idx, start_evt in start_df.iterrows():
            # end 후보: start 이후
            end_candidates = end_df[end_df['start_time'] > start_evt['start_time']]
            if end_candidates.empty:
                debug_info.append({
                    'start_time': start_evt['start_time'],
                    'status': 'no_end_candidates',
                    'message': f"No end candidates found after start at {start_evt['start_time']}"
                })
                continue
            
            # mid 이벤트 순서대로 찾기
            mid_times = []
            mid_found_status = []  # 각 mid 이벤트의 찾기 상태
            last_time = start_evt['start_time']
            found_all = True
            
            for idx, mid_df in enumerate(mid_event_dfs):
                # margin_gap 적용 (msec 단위 -> 초 변환)
                margin_gap = 0.0
                if isinstance(mid_event_cfgs[idx], dict):
                    margin_gap = float(mid_event_cfgs[idx].get('margin_gap', 0.0)) / 1000.0
                search_time = last_time + margin_gap
                
                # start~end 사이에서, 직전 mid 이후(또는 margin_gap 이후) 첫 mid
                mid_candidate = mid_df[mid_df['timestamp'] > search_time]
                if mid_candidate.empty:
                    found_all = False
                    mid_found_status.append({
                        'mid_index': idx,
                        'mid_type': mid_event_cfgs[idx]['event_type'],
                        'search_time': search_time,
                        'status': 'not_found',
                        'message': f"Mid event {idx+1} ({mid_event_cfgs[idx]['event_type']}) not found after {search_time}"
                    })
                    break
                
                mid_evt = mid_candidate.iloc[0]
                mid_times.append(mid_evt['timestamp'])
                mid_found_status.append({
                    'mid_index': idx,
                    'mid_type': mid_event_cfgs[idx]['event_type'],
                    'found_time': mid_evt['timestamp'],
                    'status': 'found'
                })
                last_time = mid_evt['timestamp']
            
            # 마지막 mid 이후 첫 end 찾기
            end_found = False
            end_time = None
            if found_all:
                end_candidate = end_candidates[end_candidates['start_time'] > last_time]
                if end_candidate.empty:
                    debug_info.append({
                        'start_time': start_evt['start_time'],
                        'status': 'no_end_after_mid',
                        'message': f"Found all mid events but no end after last mid at {last_time}",
                        'mid_found_status': mid_found_status
                    })
                else:
                    end_evt = end_candidate.iloc[0]
                    end_found = True
                    end_time = end_evt['start_time']
            else:
                # mid 이벤트를 모두 찾지 못한 경우에도 디버깅 정보 저장
                debug_info.append({
                    'start_time': start_evt['start_time'],
                    'status': 'mid_events_incomplete',
                    'message': f"Not all mid events found, found {len(mid_times)}/{len(mid_event_cfgs)}",
                    'mid_found_status': mid_found_status
                })
            
            # 결과 생성 (부분 매칭도 포함)
            if mid_times:  # 최소 하나의 mid 이벤트라도 찾았으면 결과 생성
                result = {
                    'event_name': event_config['event_name'],
                    'event_type': event_config['event_type'],
                    'start_time': start_evt['start_time'],
                    'start_tag': start_interrupt_cfg.get('tag', ''),
                    'start_event_name': start_event_name,
                    'mid_events_found': len(mid_times),
                    'total_mid_events': len(mid_event_cfgs),
                    'found_all_mid_events': found_all,
                    'end_found': end_found,
                    'end_time': end_time,
                    'end_tag': end_interrupt_cfg.get('tag', ''),
                    'end_event_name': end_event_name,
                }
                
                # mid 이벤트 시간들 저장
                for i, t in enumerate(mid_times):
                    result[f'mid_time_{i+1}'] = t
                
                # 찾지 못한 mid 이벤트들은 None으로 설정
                for i in range(len(mid_times), len(mid_event_cfgs)):
                    result[f'mid_time_{i+1}'] = None
                
                # runtime 계산 (end가 있으면 계산, 없으면 None)
                if end_found and end_time:
                    result['runtime'] = end_time - start_evt['start_time']
                else:
                    result['runtime'] = None
                
                results.append(result)
        
        result_df = pd.DataFrame(results)
        if result_df.empty:
            print(f"No complex latency event pairs found for {event_config['event_name']}")
        else:
            print(f"Found {len(result_df)} complex latency events for {event_config['event_name']}")
            # 디버깅 정보 출력
            if debug_info:
                print(f"Debug info for {len(debug_info)} failed matches:")
                for debug in debug_info[:5]:  # 처음 5개만 출력
                    print(f"  - {debug['status']}: {debug['message']}")
                if len(debug_info) > 5:
                    print(f"  ... and {len(debug_info) - 5} more failed matches")
        
        return result_df 