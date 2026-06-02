import requests, argparse, os, time
from datetime import datetime, timedelta

# 1. 根据新的cURL命令，更新导出文件的URL
URL_CHANGE_DATE = 'https://pmos.sd.sgcc.com.cn:18080/trade/main/home/changeDate.do'
URL_EXPORT_EXCEL = 'https://pmos.sd.sgcc.com.cn:18080/trade/YxJyjgydPlantQueryTmp.do?method=export'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'zh-CN,zh;q=0.9'
}

def create_session(token, sessionid):
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update({
        'Admin-Token': token,
        'X-Ticket': token,
        'XHXT_SESSIONID': sessionid,
        'Gray-Tag': '717a787831',
        'ClientTag': 'OUTNET_BROWSE',
        'CurrentRoute': '/dashboard',
        'X-Token': 'undefined'
    })
    return s

def get_start_date(folder):
    os.makedirs(folder, exist_ok=True)
    dates = []
    for f in os.listdir(folder):
        if f.endswith('.xls'):
            try:
                dates.append(datetime.strptime(f.split('_')[0], '%Y-%m-%d'))
            except:
                pass
    return max(dates) + timedelta(days=1) if dates else datetime(2026,1,5)

def safe_download(session, date_str, save_path):
    # 切换日期
    session.get(URL_CHANGE_DATE, params={'pdate': date_str, '_': int(time.time()*1000)}, timeout=30)
    
    # 2. 根据新的cURL命令，更新Referer头信息
    r = session.get(URL_EXPORT_EXCEL, headers={
        'Referer': f'https://pmos.sd.sgcc.com.cn:18080/trade/YxJyjgydPlantQueryTmp.do?appkey=217&pdate={date_str}'
    }, timeout=60)

    if r.status_code != 200 or len(r.content) < 1500:
        return False

    with open(save_path, 'wb') as f:
        f.write(r.content)
    return True

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t','--token', required=True, help='The Admin-Token and X-Ticket value.')
    # parser.add_argument('-s','--sessionid', required=True, help='The XHXT_SESSIONID value.')
    # parser.add_argument('-f','--folder', required=True, help='The folder to save downloaded files.')
    # args = parser.parse_args()

    # Credentials must be provided via environment variables
    token = os.environ.get('SGCC_TOKEN', '')
    sessionid = os.environ.get('SGCC_SESSIONID', '')
    if not token or not sessionid:
        raise RuntimeError("请设置环境变量 SGCC_TOKEN 和 SGCC_SESSIONID")
    folder = '市场运行数据用电侧'

    session = create_session(token, sessionid)
    start = get_start_date(folder)
    end = datetime.now()

    print(f'▶ 开始日期：{start.date()}')
    print(f'▶ 结束日期：{end.date()}')

    d = start
    while d <= end:
        date_str = d.strftime('%Y-%m-%d')
        file = os.path.join(folder, f'{date_str}_{folder}.xls')

        if os.path.exists(file):
            print(f'✔ 已存在 {date_str}')
            d += timedelta(days=1)
            continue

        print(f'⬇ 下载 {date_str} ...')

        success = False
        for _ in range(3):  # 自动重试
            try:
                if safe_download(session, date_str, file):
                    print(f'✔ 完成 {date_str}')
                    success = True
                    break
            except Exception as e:
                print('  重试中...', e)
                time.sleep(3)

        if not success:
            print(f'✘ {date_str} 下载失败，跳过')
        
        d += timedelta(days=1)
        time.sleep(1.2)

    print('🎉 全部下载完成')

if __name__ == '__main__':
    main()