"""Metricas de avaliacao: deteccao (P/R/F1, AP) e OCR (CER/WER)."""

from __future__ import annotations

import pytest

from lprts.eval.groundtruth import (
    GroundTruth,
    GTAnnotation,
    load_ground_truth,
    save_ground_truth,
)
from lprts.eval.metrics import (
    average_precision,
    character_error_rate,
    confusion_matrix,
    evaluate_detection,
    evaluate_ocr,
    levenshtein,
    match_detections,
    mean_average_precision,
)
from lprts.eval.report import EvaluationReport


class TestDistanciaDeEdicao:
    @pytest.mark.parametrize(
        "a,b,esperado",
        [("ABC1D23", "ABC1D23", 0), ("ABC1D23", "ABC1D28", 1), ("", "ABC", 3), ("ABC", "", 3)],
    )
    def test_levenshtein(self, a, b, esperado):
        assert levenshtein(a, b) == esperado

    def test_cer(self):
        assert character_error_rate("ABC1D23", "ABC1D28") == pytest.approx(1 / 7)

    def test_cer_referencia_vazia(self):
        assert character_error_rate("", "") == 0.0
        assert character_error_rate("", "X") == 1.0


class TestDeteccao:
    def test_casamento_respeita_iou(self):
        preds = [((10, 10, 50, 50), 0.9)]
        is_tp, _, _ = match_detections(preds, [(12, 12, 52, 52)], 0.5)
        assert is_tp == [True]

    def test_uma_referencia_casa_uma_unica_vez(self):
        """Duas predicoes sobre a mesma placa: a segunda e falso positivo."""
        preds = [((10, 10, 50, 50), 0.9), ((11, 11, 51, 51), 0.8)]
        is_tp, _, _ = match_detections(preds, [(10, 10, 50, 50)], 0.5)
        assert is_tp == [True, False]

    def test_ordem_por_confianca(self):
        """A predicao mais confiante deve ficar com a referencia."""
        preds = [((100, 100, 140, 140), 0.3), ((10, 10, 50, 50), 0.95)]
        is_tp, _, _ = match_detections(preds, [(10, 10, 50, 50)], 0.5)
        assert is_tp[0] is True  # ja reordenado por confianca

    def test_precisao_revocacao_f1(self):
        m = evaluate_detection(
            {0: [((10, 10, 50, 50), 0.9)], 1: [((0, 0, 10, 10), 0.9)]},
            {0: [(10, 10, 50, 50)], 1: [(100, 100, 140, 140)]},
            0.5,
        )
        assert m.tp == 1 and m.fp == 1 and m.fn == 1
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(0.5)

    def test_ap_perfeito(self):
        preds = {i: [((10, 10, 50, 50), 0.9)] for i in range(5)}
        gts = {i: [(10, 10, 50, 50)] for i in range(5)}
        assert evaluate_detection(preds, gts, 0.5).ap == pytest.approx(1.0, abs=1e-6)

    def test_ap_sem_predicoes(self):
        assert average_precision([], {0: [(0, 0, 10, 10)]}, 0.5) == 0.0

    def test_map_inclui_os_limiares_padrao(self):
        r = mean_average_precision({0: [((10, 10, 50, 50), 0.9)]}, {0: [(10, 10, 50, 50)]})
        assert "mAP@[.5:.95]" in r and "AP@0.50" in r and "AP@0.75" in r


class TestOCR:
    def test_exatidao_e_wer(self):
        m = evaluate_ocr([("ABC1D23", "ABC1D23"), ("XYZ9K88", "XYZ9K8B")])
        assert m.exact_accuracy == pytest.approx(0.5)
        assert m.wer == pytest.approx(0.5)

    def test_cer_agregado(self):
        m = evaluate_ocr([("ABC1D23", "ABC1D23"), ("ABC1D23", "ABC1D28")])
        assert m.cer == pytest.approx(1 / 14)

    def test_matriz_de_confusao_captura_o_erro(self):
        m = evaluate_ocr([("ABC1D23", "ABC1D2B")])
        assert ("3", "B") in m.confusions

    def test_erros_por_posicao(self):
        m = evaluate_ocr([("ABC1D23", "ABC1D2B")])
        assert m.per_position_errors[6] == 1

    def test_matriz_de_confusao_matricial(self):
        matriz, alfabeto = confusion_matrix([("AB", "AC")])
        assert matriz[alfabeto.index("A"), alfabeto.index("A")] == 1
        assert matriz[alfabeto.index("B"), alfabeto.index("C")] == 1

    def test_sem_amostras(self):
        m = evaluate_ocr([])
        assert m.n == 0 and m.exact_accuracy == 0.0 and m.cer == 0.0


class TestGroundTruth:
    @pytest.mark.parametrize("ext", [".csv", ".json"])
    def test_ida_e_volta(self, tmp_path, ext):
        gt = GroundTruth(
            video="v",
            annotations=[
                GTAnnotation(10, (1, 2, 3, 4), "abc-1d23"),
                GTAnnotation(11, (5, 6, 7, 8), "XYZ9K88", difficult=True),
            ],
        )
        caminho = save_ground_truth(gt, tmp_path / f"gt{ext}")
        recarregado = load_ground_truth(caminho)
        assert len(recarregado) == 2
        assert recarregado.annotations[0].plate_text == "ABC1D23"  # normalizado

    def test_dificeis_excluidas_por_padrao(self, tmp_path):
        gt = GroundTruth(
            annotations=[
                GTAnnotation(1, (0, 0, 1, 1), "ABC1D23"),
                GTAnnotation(2, (0, 0, 1, 1), "XYZ9K88", difficult=True),
            ]
        )
        assert gt.unique_plates() == {"ABC1D23"}
        assert len(gt.unique_plates(include_difficult=True)) == 2

    def test_arquivo_ausente(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ground_truth(tmp_path / "nao_existe.csv")

    def test_indice_por_quadro(self):
        gt = GroundTruth(
            annotations=[
                GTAnnotation(5, (0, 0, 1, 1), "ABC1D23"),
                GTAnnotation(5, (2, 2, 3, 3), "XYZ9K88"),
            ]
        )
        assert len(gt.by_frame()[5]) == 2


class TestRelatorio:
    def test_texto_json_e_tabelas(self, tmp_path):
        rel = EvaluationReport(
            video="v.mp4",
            n_frames=100,
            detection=evaluate_detection({0: [((0, 0, 10, 10), 0.9)]}, {0: [(0, 0, 10, 10)]}),
            ocr_frame_level=evaluate_ocr([("ABC1D23", "ABC1D28")]),
            frame_level={"accuracy": 0.0, "accuracy_ci": [0.0, 0.0], "rejection_rate": 0.0},
            vehicle_level={"n_vehicles": 1, "recall": 1.0, "recall_ci": [1.0, 1.0],
                           "precision": 1.0, "precision_ci": [1.0, 1.0], "f1": 1.0},
            fusion_comparison={"n": 1, "single_read": 0.5, "best_read": 1.0, "fused": 1.0,
                               "published": 1.0},
            vehicles=[{"vehicle": "id:1", "published_correct": 1}],
        )
        texto = rel.to_text()
        assert "RELATORIO DE AVALIACAO" in texto
        assert "POR VEICULO" in texto
        assert "EFEITO DA FUSAO" in texto      # comparacao na mesma populacao
        caminho = rel.save_json(tmp_path / "r.json")
        assert caminho.exists()
        tabelas = rel.save_tables(caminho)
        assert any(t.name == "r_veiculos.csv" and t.stat().st_size > 0 for t in tabelas)
