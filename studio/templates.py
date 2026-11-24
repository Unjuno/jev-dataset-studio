"""Small public examples; all sample sentences are invented for this project."""


SAMPLE = [
    {"id": "note-001", "equipment": "pump-A", "text": "ポンプAで振動が増加した。点検を実施し、軸受の交換を検討する。"},
    {"id": "note-002", "equipment": "pump-A", "text": "ポンプAの振動は正常範囲だった。発熱は認められない。"},
    {"id": "note-003", "equipment": "motor-B", "text": "モーターBが発熱したため停止した。振動の有無はまだ確認できていない。"},
    {"id": "note-004", "equipment": "valve-C", "text": "バルブCを目視点検した。異常はないとの報告だが、測定記録が不足している。"},
    {"id": "note-005", "equipment": "sensor-D", "text": "センサーD🔧の信号を確認した。信号は一度途切れたが、その後は安定している。"},
    {"id": "note-006", "equipment": "fan-E", "text": "ファンEで異音が発生した。異音は清掃後も続いた。再点検が必要だ。"},
]


def builtin_templates(model="jev-1.14.0"):
    from .schemas import Recipe
    from .exports import ExportSpec
    equipment = Recipe.model_validate({
        "name": "設備点検のラベルと重要箇所", "provider": "demo", "model": model,
        "source": {"text_field": "text", "id_field": "id", "group_field": "equipment"},
        "target": {"mode": "sentence"}, "context": {"mode": "window", "window_chars": 180},
        "features": [
            {"id": "phenomenon", "name": "現象", "kind": "choice",
             "instruction": "対象の文が述べる主な現象を文脈に基づいて分類する。未確認、否定、正常の記述も現象の種類とは別に読む。",
             "choices": {"vibration": "振動に関する記述", "heat": "発熱・温度に関する記述",
                         "sound": "異音に関する記述", "other": "その他・特定できない"}},
            {"id": "observed", "name": "異常の確認", "kind": "boolean",
             "instruction": "対象文は異常が実際に確認されたことを述べているか。否定、未確認、正常範囲は肯定しない。"},
            {"id": "importance", "name": "対応の重要度", "kind": "score",
             "instruction": "対象文が設備への対応判断に与える重要性を、明示された事実だけに基づいて評価する。",
             "levels": ["通常の記録・対応不要", "確認や経過観察が必要", "停止・交換・再点検などの対応が必要"]},
        ],
    })
    generic = Recipe.model_validate({
        "name": "自由な文章分類", "provider": "demo", "model": model,
        "source": {"text_field": "text", "id_field": None, "group_field": None},
        "target": {"mode": "document"}, "context": {"mode": "document"},
        "features": [{"id": "label", "name": "分類ラベル", "kind": "choice",
                      "instruction": "文章が具体的な事実の観測、意見・提案、または判断できない内容のどれを主に述べるか分類する。",
                      "choices": {"observation": "事実の観測や報告", "opinion": "意見や提案", "unknown": "判断できない"}}],
    })
    templates = []
    for key, recipe in [("builtin-equipment", equipment), ("builtin-generic", generic)]:
        export = ExportSpec.model_validate({"columns": [
            {"name": "source_id", "source": "source_id", "role": "metadata"},
            {"name": "text", "source": "target.text", "role": "input"},
            *[{"name": f.id, "source": "features." + f.id, "role": "target"} for f in recipe.features],
            {"name": "origin", "source": "origin", "role": "metadata"},
            {"name": "reviewed", "source": "reviewed", "role": "metadata"},
        ]})
        templates.append({"id": key, "name": recipe.name, "recipe": recipe.model_dump(mode="json"),
                          "export_spec": export.model_dump(mode="json"), "builtin": True})
    return templates
