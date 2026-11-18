"""Limpeza dos dados anteriores no inicio de cada processamento."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from lprts.analytics.store import ResultStore, clear_results
from lprts.config import AppConfig
from lprts.factory import clear_previous_results

RAIZ = Path(__file__).resolve().parent.parent
SINTETICO = RAIZ / "data" / "videos" / "sintetico.mp4"


def _banco_com_sessoes(db: Path, n: int = 2) -> list[str]:
    ids = []
    for _ in range(n):
        with ResultStore(db) as store:
            store.start_session("v.mp4")
            ids.append(store.session_id)
    return ids


class TestClearResults:
    def test_apaga_o_banco_e_so_os_recortes_das_sessoes_dele(self, tmp_path):
        db = tmp_path / "r.db"
        s1, s2 = _banco_com_sessoes(db)
        Path(str(db) + "-wal").write_bytes(b"")
        crops = tmp_path / "crops"
        crops.mkdir()
        dele = [crops / f"{s1}_t1_f10_ABC1D23.jpg", crops / f"{s2}_t4_f2_naopublicada.jpg"]
        # Recortes de outro banco (a pasta e compartilhada com o da avaliacao) e
        # arquivos que nao sao recortes ficam.
        outros = [crops / "a1b2c3d4e5f6_t1_f1_XYZ9K88.jpg", crops / f"{s1}_notas.txt"]
        for arq in dele + outros:
            arq.write_bytes(b"x")

        resumo = clear_results(db, crops)

        assert resumo.db_removed and resumo.n_sessions == 2 and resumo.n_crops == 2
        assert not db.exists() and not Path(str(db) + "-wal").exists()
        assert not any(a.exists() for a in dele)
        assert all(a.exists() for a in outros)

    def test_sem_banco_nao_faz_nada(self, tmp_path):
        resumo = clear_results(tmp_path / "nao_existe.db", tmp_path)
        assert not resumo.db_removed
        assert resumo.describe().startswith("Nenhum dado anterior")
        assert str(tmp_path / "nao_existe.db") in resumo.describe()   # caminho intacto

    @pytest.mark.parametrize("tipo", ["sqlite_de_outro_programa", "texto"])
    def test_nao_apaga_arquivo_que_nao_e_banco_do_lprts(self, tmp_path, tipo):
        alvo = tmp_path / "importante.db"
        if tipo == "texto":
            alvo.write_text("dados de outro programa")
        else:
            with sqlite3.connect(alvo) as con:
                con.execute("CREATE TABLE clientes (nome TEXT)")
        with pytest.raises(ValueError, match="nao e um banco do LPR-RTS"):
            clear_results(alvo, tmp_path)
        assert alvo.exists()

    def test_banco_vazio_e_apagado(self, tmp_path):
        db = tmp_path / "vazio.db"
        db.write_bytes(b"")
        assert clear_results(db).db_removed


class TestConfiguracao:
    def test_padrao_limpa_e_nao_muda_o_hash_da_configuracao(self):
        a, b = AppConfig(), AppConfig()
        b.analytics.clear_previous = False
        assert a.analytics.clear_previous is True
        assert a.fingerprint() == b.fingerprint()

    def test_desligada_preserva_o_banco(self, tmp_path):
        db = tmp_path / "r.db"
        _banco_com_sessoes(db, 1)
        cfg = AppConfig()
        cfg.analytics.db_path = str(db)
        cfg.analytics.clear_previous = False
        assert clear_previous_results(cfg) is None
        assert db.exists()


@pytest.mark.skipif(not SINTETICO.exists(), reason="video sintetico ausente")
def test_eval_limpa_uma_vez_e_mantem_as_repeticoes_juntas(tmp_path):
    from typer.testing import CliRunner

    from lprts.cli import app

    perfil = tmp_path / "saida.yaml"
    perfil.write_text(f"analytics:\n  crops_dir: {tmp_path / 'crops'}\n", encoding="utf-8")
    db = tmp_path / "eval.db"
    base = ["eval", "--source", str(SINTETICO), "--gt", str(SINTETICO.with_name("sintetico_gt.csv")),
            "--config", str(perfil), "--no-vehicle-detector", "--plate-detector", "classic",
            "--ocr", "template", "--max-frames", "12", "--db", str(db),
            "--out", str(tmp_path / "r.json")]

    def sessoes() -> int:
        with sqlite3.connect(db) as con:
            return con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    r = CliRunner().invoke(app, [*base, "--repeat", "2"])
    assert r.exit_code == 0, r.output
    assert sessoes() == 2                       # as repeticoes ficam juntas
    r = CliRunner().invoke(app, base)
    assert r.exit_code == 0, r.output
    assert sessoes() == 1                       # o novo processamento comeca do zero
    r = CliRunner().invoke(app, [*base, "--keep-previous"])
    assert r.exit_code == 0, r.output
    assert sessoes() == 2


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="interface grafica nao instalada")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


class TestInterface:
    def test_tipos_de_objeto_e_limpeza_vao_para_a_configuracao(self, qapp, tmp_path):
        import importlib.util

        from lprts.ui.app import MainWindow, _selecionar

        if importlib.util.find_spec("ultralytics") is None:
            pytest.skip("ultralytics nao instalado: a caixa de veiculos fica desabilitada")
        pesos = tmp_path / "yolo.pt"
        pesos.write_bytes(b"x")
        cfg = AppConfig()
        cfg.source.uri = str(tmp_path / "IMG_0834.MOV")
        cfg.detector.vehicle_weights = str(pesos)
        janela = MainWindow(cfg)
        try:
            assert janela.clear_check.isChecked()
            assert janela.classes_combo.isEnabled()
            assert _selecionar(janela.classes_combo, "completo")
            janela.clear_check.setChecked(False)
            janela._apply_controls()
            assert cfg.detector.classes == "completo"
            assert cfg.analytics.clear_previous is False
            # Sem o YOLO nao ha tipos de objeto a escolher.
            janela.vehicle_check.setChecked(False)
            assert not janela.classes_combo.isEnabled()
        finally:
            janela.close()

    def test_resumo_de_objetos(self, qapp):
        from lprts.ui.widgets import resumo_de_objetos

        assert resumo_de_objetos({}) == "-"
        assert resumo_de_objetos({"pessoa": 3, "carro": 12, "bicicleta": 3}) == (
            "carro 12 · bicicleta 3 · pessoa 3")
        muitos = {f"c{i}": 1 for i in range(8)}
        assert resumo_de_objetos(muitos, maximo=6).endswith("· +2")
