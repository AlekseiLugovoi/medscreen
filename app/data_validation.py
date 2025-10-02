# --- Модуль для валидации данных исследования ---
# Флоу:
# 1. `validate_series` получает на вход метаданные одной серии.
# 2. Проводит набор проверок (модальность, ориентация, кол-во срезов).
# 3. Возвращает список словарей, где каждый словарь - результат одной проверки.
#    `[{"check": "Название", "status": True/False, "message": "Сообщение"}]`

def validate_series(meta: dict) -> list:
    """Проводит валидацию серии по заданным критериям."""
    checks = []
    
    # 1. Проверка формата источника
    source_format = meta.get("SourceFormat", "N/A")
    is_valid_source = source_format != "N/A"
    checks.append({
        "check": "Формат источника",
        "status": is_valid_source,
        "message": f"Обнаружен формат: {source_format}"
    })

    # 2. Проверка модальности
    modality = meta.get("Modality", "N/A")
    is_valid_modality = modality in ["CT", "NIFTI"]
    checks.append({
        "check": "Модальность (КТ)",
        "status": is_valid_modality,
        "message": f"Обнаружена модальность: {modality}"
    })

    # 3. Проверка области исследования (Body Part)
    body_part = meta.get("BodyPartExamined", "N/A").upper()
    # Ищем ключевые слова, относящиеся к грудной клетке
    is_chest = any(keyword in body_part for keyword in ["CHEST", "THORAX", "LUNG", "ГРУД"])
    checks.append({
        "check": "Область (Грудная клетка)",
        "status": is_chest,
        "message": f"Указанная область: {meta.get('BodyPartExamined', 'N/A')}"
    })

    # 4. Проверка ориентации
    orientation = meta.get("orientation", "Unknown")
    is_axial = orientation == "Axial"
    checks.append({
        "check": "Ориентация (Аксиальная)",
        "status": is_axial,
        "message": f"Определена ориентация: {orientation}"
    })

    # 5. Проверка количества срезов
    num_frames = meta.get("num_frames", 0)
    is_enough_slices = num_frames > 10
    checks.append({
        "check": "Количество срезов (> 10)",
        "status": is_enough_slices,
        "message": f"Найдено срезов: {num_frames}"
    })

    return checks