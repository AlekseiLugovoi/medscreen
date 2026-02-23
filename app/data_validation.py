def validate_series(meta: dict) -> list:
    """Validate series metadata against screening requirements."""
    checks = []

    source_format = meta.get("SourceFormat", "N/A")
    checks.append({
        "check": "Source Format",
        "status": source_format != "N/A",
        "message": f"Detected: {source_format}",
    })

    modality = meta.get("Modality", "N/A")
    checks.append({
        "check": "Modality (CT)",
        "status": modality in ["CT", "NIFTI"],
        "message": f"Detected: {modality}",
    })

    body_part = meta.get("BodyPartExamined", "N/A").upper()
    is_chest = any(kw in body_part for kw in ["CHEST", "THORAX", "LUNG"])
    is_unknown = body_part in ["N/A", ""]
    checks.append({
        "check": "Body Part (Chest)",
        "status": is_chest or is_unknown,
        "message": f"Detected: {meta.get('BodyPartExamined', 'N/A')}" + (" (unknown, assumed OK)" if is_unknown else ""),
    })

    orientation = meta.get("orientation", "Unknown")
    checks.append({
        "check": "Orientation (Axial)",
        "status": orientation == "Axial",
        "message": f"Detected: {orientation}",
    })

    num_frames = meta.get("num_frames", 0)
    checks.append({
        "check": "Slice Count (> 10)",
        "status": num_frames > 10,
        "message": f"Found: {num_frames} slices",
    })

    return checks
