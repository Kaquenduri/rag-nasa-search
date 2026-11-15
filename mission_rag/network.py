import functools
import ssl
from urllib.request import HTTPSHandler

import truststore


@functools.lru_cache(maxsize=1)
def tls_context():
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def https_handler():
    return HTTPSHandler(context=tls_context())
