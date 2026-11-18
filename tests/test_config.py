"""Configuracao: carga, gravacao, sobreposicao e reconstrucao aninhada."""

from __future__ import annotations

import pytest

from lprts.config import AnalyticsConfig, AppConfig


class TestAppConfig:
    def test_padroes_sensatos(self):
        c = AppConfig()
        assert c.source.realtime is False           # avaliacao e o padrao seguro
        assert c.fusion.require_valid_syntax is True
        assert c.detector.device == "auto"

    def test_ida_e_volta_em_yaml(self, tmp_path):
        c = AppConfig()
        c.ocr.engine = "paddle"
        c.detector.device = "coreml"
        c.source.realtime = True
        caminho = c.save(tmp_path / "c.yaml")
        recarregado = AppConfig.load(caminho)
        assert recarregado.ocr.engine == "paddle"
        assert recarregado.detector.device == "coreml"
        assert recarregado.source.realtime is True

    def test_reconstroi_dataclasses_aninhadas(self):
        c = AppConfig.from_dict({"analytics": {"db_path": "x.db", "save_crops": False}})
        assert isinstance(c.analytics, AnalyticsConfig)
        assert c.analytics.db_path == "x.db" and c.analytics.save_crops is False

    def test_override_com_caminho_pontilhado(self):
        c = AppConfig().override(**{"ocr.engine": "easyocr", "source.max_frames": 100})
        assert c.ocr.engine == "easyocr" and c.source.max_frames == 100

    def test_override_ignora_none(self):
        """Permite repassar argumentos opcionais da CLI sem filtrar."""
        c = AppConfig()
        original = c.detector.conf
        c.override(**{"detector.conf": None})
        assert c.detector.conf == original

    def test_override_desconhecido_levanta(self):
        with pytest.raises(AttributeError):
            AppConfig().override(**{"ocr.inexistente": 1})

    def test_arquivo_ausente_levanta(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AppConfig.load(tmp_path / "nao_existe.yaml")

    def test_sem_caminho_devolve_padroes(self):
        assert AppConfig.load(None).ocr.engine == "auto"

    @pytest.mark.parametrize("perfil", ["default", "tempo_real", "tempo_real_rapido", "avaliacao", "lote"])
    def test_perfis_do_repositorio_carregam(self, perfil):
        """Os YAMLs versionados precisam continuar validos."""
        from pathlib import Path

        caminho = Path(__file__).resolve().parent.parent / "configs" / f"{perfil}.yaml"
        if not caminho.exists():
            pytest.skip(f"perfil ausente: {caminho}")
        c = AppConfig.load(caminho)
        assert c.ocr.engine
        assert c.analytics.db_path

    @pytest.mark.parametrize(("variante", "opcao", "valor"), [
        ("sem_recortes", "analytics.save_crops", False),
        ("o1_ultralytics", "detector.runtime", "ultralytics"),
        ("o2_todas_placas", "detector.plate_width_ratio", 0),
        ("o3_recorte_reduzido", "ocr.full_resolution", False),
        ("o4_ffmpeg", "source.backend", "ffmpeg"),
        ("res_960", "source.resize_width", 960),
        ("res_1920", "source.resize_width", 1920),
    ])
    def test_variante_do_artigo_muda_so_a_sua_opcao(self, variante, opcao, valor):
        """Cada YAML de configs/artigo desliga uma coisa so sobre o perfil de tempo real."""
        from dataclasses import asdict
        from pathlib import Path

        configs = Path(__file__).resolve().parent.parent / "configs"
        base = AppConfig.load(configs / "tempo_real.yaml")
        com = AppConfig.load_many([configs / "tempo_real.yaml", configs / "artigo" / f"{variante}.yaml"])
        secao, campo = opcao.split(".")
        assert getattr(getattr(com, secao), campo) == valor
        assert getattr(getattr(base, secao), campo) != valor      # a variante muda de fato o perfil
        antes, depois = asdict(base), asdict(com)
        depois[secao][campo] = antes[secao][campo]
        assert depois == antes
