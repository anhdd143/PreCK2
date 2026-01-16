"""
Cấu hình hệ thống phân tích kỹ thuật chứng khoán Việt Nam
"""
import os
from datetime import datetime, timedelta

# ============================================================
# DANH SÁCH MÃ CỔ PHIẾU MẶC ĐỊNH
# ============================================================
DEFAULT_SYMBOLS = [
    "VCK", "DGW", "VNM", "FPT", "VIC", 
    "HPG", "MWG", "TCB", "VCB", "ACB", "VPB"
]

# ============================================================
# CẤU HÌNH LƯU TRỮ
# ============================================================
BASE_OUTPUT_DIR = "./output"

# ============================================================
# DANH SÁCH 26 CHỈ BÁO KỸ THUẬT
# ============================================================
INDICATORS_CONFIG = {
    # Nhóm XU HƯỚNG (Trend) - 9 chỉ báo
    "SMA_5": {"group": "Xu hướng", "name": "SMA 5", "enabled": True, "params": {"period": 5}},
    "SMA_10": {"group": "Xu hướng", "name": "SMA 10", "enabled": True, "params": {"period": 10}},
    "SMA_20": {"group": "Xu hướng", "name": "SMA 20", "enabled": True, "params": {"period": 20}},
    "SMA_50": {"group": "Xu hướng", "name": "SMA 50", "enabled": True, "params": {"period": 50}},
    "SMA_200": {"group": "Xu hướng", "name": "SMA 200", "enabled": False, "params": {"period": 200}},
    "EMA_12": {"group": "Xu hướng", "name": "EMA 12", "enabled": True, "params": {"period": 12}},
    "EMA_26": {"group": "Xu hướng", "name": "EMA 26", "enabled": True, "params": {"period": 26}},
    "EMA_50": {"group": "Xu hướng", "name": "EMA 50", "enabled": False, "params": {"period": 50}},
    "ADX": {"group": "Xu hướng", "name": "ADX", "enabled": True, "params": {"period": 14}},
    
    # Nhóm ĐỘNG LƯỢNG (Momentum) - 4 chỉ báo
    "MACD": {"group": "Động lượng", "name": "MACD (12,26,9)", "enabled": True, "params": {"fast": 12, "slow": 26, "signal": 9}},
    "RSI": {"group": "Động lượng", "name": "RSI 14", "enabled": True, "params": {"period": 14}},
    "ROC": {"group": "Động lượng", "name": "ROC 10", "enabled": True, "params": {"period": 10}},
    "MOM": {"group": "Động lượng", "name": "Momentum 10", "enabled": True, "params": {"period": 10}},
    
    # Nhóm DAO ĐỘNG (Oscillator) - 5 chỉ báo
    "STOCH": {"group": "Dao động", "name": "Stochastic (14,3)", "enabled": True, "params": {"k_period": 14, "d_period": 3}},
    "BB": {"group": "Dao động", "name": "Bollinger Bands (20,2)", "enabled": True, "params": {"period": 20, "std": 2}},
    "ATR": {"group": "Dao động", "name": "ATR 14", "enabled": True, "params": {"period": 14}},
    "CCI": {"group": "Dao động", "name": "CCI 20", "enabled": True, "params": {"period": 20}},
    "WILLIAMS_R": {"group": "Dao động", "name": "Williams %R 14", "enabled": True, "params": {"period": 14}},
    
    # Nhóm KHỐI LƯỢNG (Volume) - 8 chỉ báo
    "OBV": {"group": "Khối lượng", "name": "OBV", "enabled": True, "params": {}},
    "MFI": {"group": "Khối lượng", "name": "MFI 14", "enabled": True, "params": {"period": 14}},
    "CMF": {"group": "Khối lượng", "name": "CMF 20", "enabled": True, "params": {"period": 20}},
    "VWAP": {"group": "Khối lượng", "name": "VWAP", "enabled": False, "params": {}},
    "VOL_RATIO": {"group": "Khối lượng", "name": "Volume Ratio", "enabled": True, "params": {"period": 20}},
    "VOL_SMA": {"group": "Khối lượng", "name": "Volume SMA 20", "enabled": True, "params": {"period": 20}},
    "AD": {"group": "Khối lượng", "name": "A/D Line", "enabled": True, "params": {}},
    "FORCE_INDEX": {"group": "Khối lượng", "name": "Force Index 13", "enabled": False, "params": {"period": 13}},
}

# ============================================================
# CHỈ SỐ THỊ TRƯỜNG
# ============================================================
MARKET_INDICES = {
    "VNINDEX": "VN-Index",
    "VN30": "VN30", 
    "HNX30": "HNX30",
}

# ============================================================
# CẤU HÌNH DỰ BÁO
# ============================================================
FORECAST_PERIODS = {
    "short": ["T1", "T2", "T3", "T4", "T5"],      # 1-5 ngày
    "medium": ["W1", "W2", "W3", "W4"],            # 1-4 tuần
    "long": ["M1", "M2", "M3"],                    # 1-3 tháng
}
