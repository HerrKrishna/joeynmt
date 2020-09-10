import sys
import re

if __name__ == '__main__':

    input_file = sys.argv[1]

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    

    per_sec_pattern = re.compile('Tokens per Sec: +(\d+),')
    duration_pattern = re.compile('duration: +(\d+\.\d+)s')
    per_sec_total = 0
    per_sec_count = 0
    duration_total = 0
    duration_count = 0
    for line in lines:
        per_sec_match = per_sec_pattern.search(line)
        duration_match = duration_pattern.search(line)
        if per_sec_match:
            per_sec_total += float(per_sec_match.group(1))
            per_sec_count += 1
        if duration_match:
            duration_total += float(duration_match.group(1))
            duration_count += 1
        
    per_sec_average = per_sec_total/per_sec_count
    duration_average = duration_total/duration_count
    print('average tokens per sec:', per_sec_average)
    print('average duration per val:', duration_average)




