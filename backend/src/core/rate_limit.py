from slowapi import Limiter


def get_client_ip(request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_client_ip, default_limits=["120/minute"])
