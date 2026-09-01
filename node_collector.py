import requests
import base64
import socket
import re
import json
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

print(">>> 极品挖掘机 7.0：百台引擎全开版启动 <<<")

# 修正：定义原始频道列表，确保变量名一致
RAW_CHANNELS = [
    "v2ray_configs_pool", "ip_cf_config", "freakconfig", "oneclickvpnkeys", "privatevpns",
    "directvpn", "vlessconfig", "manvpn", "eliv2ray", "outline_vpn", "ppt_f66_zhk2zdy8",
    "v2rayngx", "ccbaohe", "wangcai_8", "vpn_3000", "academi_vpn", "dingyue_center",
    "freedatazone1", "freev2rayi", "mypremium98", "inikotesla", "v2rayngalpha",
    "v2rayngalphagamer", "jiedian_share", "vpn_mafia", "dr_v2ray", "allv2board",
    "bigsmoke_config", "vpn_443", "prossh", "mftizi", "qun521", "v2rayng_my2",
    "go4sharing", "trand_farsi", "vpnplusee_free", "freekankan", "awxdy666",
    "pgkj666", "anranbp", "hkaa0", "wxgqlfx", "freevpnjd", "arzhecn", "schpd",
    "jichang_list", "linux_do_channel", "nodeseekc", "hostloc_pro", "serveruniverse",
    "sharecentrepro", "impart_cloud", "helingqi", "ai_news_cn", "newlearner",
    "docofcard", "baipiao_ml", "jichangtuijian", "airport_news", "freemason6",
    "jichangbaipiao", "v2free666", "v2rayfree", "mianfeifq", "ssr_v2ray_clash",
    "free_v2ray", "node_share", "freenode_all", "baipiaonote", "jichang_store",
    "vpnoff", "jichang0", "v2list", "ssr_clash_v2ray", "clash_node_share",
    "free_jichang", "getfreenode", "v2ray_vpn_free", "v2ray_free_config", "vpn_node_share",
    "free_config_sharing", "v2ray_sub_share", "free_sub_links", "clash_config", "v2ray_pool",
    "free_node_sharing", "v2ray_nodes_free", "jichang_bp", "sub_share_center",
    "ss_v2ray_ssr_clash", "daily_free_nodes", "proxy_sharing", "fast_v2ray",
    "free_internet_news", "node_collection", "clash_v2ray_ssr_nodes", "v2ray_helper",
    "free_proxy_list", "open_v2ray", "clash_sub", "v2ray_best", "node_king",
    "free_server_list", "v2ray_pro_free", "jichang_report", "free_node_daily",

    # 高频更新
    "v2raydailyupdate", "v2rayn_node", "v2ray_share", "freevpn8",
    "v2ray_freedom", "v2raynodeshare", "v2raynconfig", "ssrtool",
    "proxy_kun", "clashnode", "nicevpn", "freeclashnode",
    "vless_vmess_trojan", "v2rayng_fast", "v2rayng_best",
    "v2rayng_daily", "v2rayng_config", "v2rayng_sub",
    "clashnodeupdate", "clashconfigdaily", "clashmeta_free",
    "freevpn365", "freevpncloud", "freevpnplanet",

    # 中频更新
    "ssrlist", "ssrconfig", "clashconfig", "clashfree",
    "vpnfailover", "v2board_sub", "v2rayngplus",
    "v2raynghub", "v2rayngcenter", "v2rayngcloud",
    "clashnodehub", "clashnodecloud", "clashnodecenter",
    "v2raynodeupdate", "v2raynodecloud", "v2raynodecenter",
    "freev2raysub", "freev2raydaily", "freev2raycloud",

    # 低频但高质量
    "vpnplus", "vpnfreeplus", "vpnconfigplus",
    "v2raypremium", "v2rayelite", "v2rayvip",
    "clashpremium", "clashelite", "clashvip",
    "vlesspremium", "trojanpremium", "sspremium",

    # 国际节点频道
    "global_vpn_free", "usa_vpn_free", "japan_vpn_free",
    "korea_vpn_free", "europe_vpn_free", "singapore_vpn_free",
    "hk_vpn_free", "tw_vpn_free", "ca_vpn_free",

    # 爬虫友好频道
    "nodefree", "nodefree2", "nodefree3",
    "clashnodefree", "clashnodefree2",
    "v2raynodefree", "v2raynodefree2",
    "subnodefree", "subnodefree2",

    # 机场分享频道
    "airport_free", "airport_share", "airport_club",
    "airport_daily", "airport_best", "airport_sub",
    "airport_node", "airport_config",

    # 订阅源发布频道
    "subconverter", "subconverter_update", "subconverter_config",
    "subconverter_node", "subconverter_clash",

    # 技术类频道
    "linux_china", "linux_tech", "linuxhub",
    "cloudflare_tech", "cftech", "cfupdate",
    "proxytech", "networktech", "internettech"
]

TG_CHANNELS = list(set([c.lower() for c in RAW_CHANNELS]))

SOURCES = [
    "https://raw.githubusercontent.com/yebekhe/TelegramV2rayCollector/main/sub/base64/mix",
    "https://raw.githubusercontent.com/wzdnzd/aggregator/main/subscribe/proxy.txt",
    "https://raw.githubusercontent.com/vfarid/v2ray-share/main/all.txt",
    "https://raw.githubusercontent.com/freefq/free/master/v2",
    "https://raw.githubusercontent.com/Pawpieee/Free-Proxies/main/sub/sub_merge.txt"
]

def safe_decode(data):
    if not data: return ""
    data = data.strip().replace('\n', '').replace('\r', '')
    try:
        for _ in range(3):
            try: return base64.b64decode(data).decode('utf-8', errors='ignore')
            except: data += '='
        return ""
    except: return ""

def extract_nodes(content):
    # 修正点：将 (vmess|vless|...) 改为 (?:vmess|vless|...)
    # 这样 findall 会返回完整的匹配字符串内容，而不仅仅是协议头
    pattern = r'(?:vmess|vless|ss|ssr|trojan|hysteria2|hy2|tuic|wireguard)://[a-zA-Z0-9%=\+\?\.\/\-\_\:\@\#\&]+'
    return re.findall(pattern, content)

def get_host_port(node):
    try:
        if node.startswith("vmess://"):
            data = json.loads(safe_decode(node[8:]))
            return data.get('add'), data.get('port')
        parsed = urlparse(node)
        host_port = parsed.netloc.split('@')[-1] if '@' in parsed.netloc else parsed.netloc
        if ':' in host_port:
            h, p = host_port.split(':')
            return h.strip(), p.strip()
    except: pass
    return None, None

def check_tcp(node):
    # 插入逻辑：对于 UDP 类协议（无法通过简单 TCP 探测的），直接跳过测试并返回 node 予以保存
    udp_protocols = ["hysteria2://", "hy2://", "tuic://", "wireguard://"]
    if any(node.startswith(p) for p in udp_protocols):
        return node
        
    h, p = get_host_port(node)
    if not h or not p: return None
    try:
        with socket.create_connection((h, int(p)), timeout=2):
            return node
    except:
        return None

def main():
    raw_pool = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0'}

    print(f"\n--- [阶段 1] 正在横扫 {len(TG_CHANNELS)} 个极品频道 ---")
    for i, channel in enumerate(TG_CHANNELS):
        try:
            if i % 10 == 0 and i > 0: time.sleep(1)
            
            r = requests.get(f"https://t.me/s/{channel}", headers=headers, timeout=10)
            if r.status_code == 200:
                found = extract_nodes(r.text)
                if found:
                    print(f"  [{i+1:03d}] @{channel.ljust(20)} | 发现: {len(found)}")
                    raw_pool.extend(found)
            else:
                print(f"  [{i+1:03d}] @{channel.ljust(20)} | 状态: {r.status_code}")
        except: continue

    print(f"\n--- [阶段 2] 正在提取核心聚合源 ---")
    for url in SOURCES:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            # 兼容直接抓取和 Base64 解码后的抓取
            found = extract_nodes(r.text) or extract_nodes(safe_decode(r.text))
            print(f"  [源] {url.split('/')[-1].ljust(20)} | 发现: {len(found)}")
            raw_pool.extend(found)
        except: continue

    # 逻辑插入：在此处保存一份未去重、未测试的集合版，用于故障排查
    if raw_pool:
        with open("all_collected.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(raw_pool))
        print(f"\n[调试说明] 原始全量数据已保存至: all_collected.txt (总数: {len(raw_pool)})")

    unique_nodes = list(set(raw_pool))
    print(f"\n--- [阶段 3] 挖掘完毕 | 原始总数: {len(raw_pool)} | 去重后: {len(unique_nodes)} ---")

    if not unique_nodes:
        print("!!! 本轮未抓取到任何节点，请检查网络 !!!")
        return

    print("正在进行精选探活 (目标 2000 个，启用多线程)...")
    final_nodes = []
    seen_ips = set()
    
    # 极品协议排序
    unique_nodes.sort(key=lambda x: ("reality" in x.lower() or "vless" in x.lower() or "trojan" in x.lower() or "hy" in x.lower()), reverse=True)

    # 逻辑插入：使用 ThreadPoolExecutor 进行高效探活
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_tcp, node): node for node in unique_nodes}
        
        for future in as_completed(futures):
            result_node = future.result()
            if result_node:
                h, p = get_host_port(result_node)
                # 如果是直接跳过测试的 UDP 节点，h 为 None 的概率较大，需特殊处理确保不被去重机制误伤
                if not h or h not in seen_ips:
                    final_nodes.append(result_node)
                    if h: seen_ips.add(h)
                    
                    if len(final_nodes) % 20 == 0:
                        print(f"  [+] 已捕获 {len(final_nodes)} 个可用极品节点")
                
            if len(final_nodes) >= 2000:
                break

    if final_nodes:
        # 输出完整订阅内容
        out = base64.b64encode("\n".join(final_nodes).encode('utf-8')).decode('utf-8')
        with open("nodes.txt", "w") as f:
            f.write(out)
        # 同时保存一个明文版方便查看
        with open("nodes_raw.txt", "w") as f:
            f.write("\n".join(final_nodes))
        print(f"\n任务成功！已将 {len(final_nodes)} 个极品节点保存至 nodes.txt")
        print("明文链接已保存至 nodes_raw.txt")
    else:
        print("\n很遗憾，本轮抓取的节点全部失效。")

if __name__ == "__main__":
    main()
