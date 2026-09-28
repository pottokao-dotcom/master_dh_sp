"""範例照網頁伺服器:多執行緒、排隊上限 128(python -m http.server 只有 5,手機一次抓幾十張縮圖會被 reset)。"""
import functools, http.server, sys
class S(http.server.ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[2])
S(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
