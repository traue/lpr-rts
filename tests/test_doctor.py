"""Diagnostico do ambiente."""

from __future__ import annotations

from lprts import doctor
from lprts.doctor import AVISO, ERRO, OK, Check, proximo_passo, resumo, run_all


class TestVerificacoes:
    def test_python_suportado(self):
        c = doctor.check_python()
        assert c.estado in (OK, AVISO)      # o interpretador de teste ja e valido

    def test_nucleo_presente(self):
        """numpy, OpenCV e typer sao obrigatorios; se faltam, os testes nem rodam."""
        assert all(c.estado == OK for c in doctor.check_nucleo())

    def test_pipeline_monta_e_processa(self):
        assert doctor.check_pipeline().estado == OK

    def test_escrita_verificada(self):
        assert doctor.check_escrita().estado in (OK, AVISO)

    def test_ocr_sempre_disponivel(self):
        """O baseline interno garante que sempre ha um motor."""
        c = doctor.check_ocr()
        assert c.estado in (OK, AVISO)
        assert "template" in c.detalhe or c.estado == OK

    def test_camera_nao_e_aberta_sem_pedir(self):
        """Abrir a webcam sem o usuario pedir dispara pedido de permissao no macOS."""
        assert doctor.check_camera(testar=False).detalhe == "nao testada"

    def test_dependencias_opcionais_nunca_sao_erro(self):
        """Interface, deep learning e analise sao opcionais - no maximo aviso."""
        opcionais = [
            doctor.check_interface(), doctor.check_ocr(),
            doctor.check_analise(), doctor.check_paralelismo(),
            *doctor.check_deep_learning(), *doctor.check_modelos(),
        ]
        assert all(c.estado != ERRO for c in opcionais)


class TestRelatorio:
    def test_roda_tudo(self):
        checks = run_all(rapido=True)
        assert len(checks) > 10
        assert all(isinstance(c, Check) for c in checks)

    def test_resumo_soma_o_total(self):
        checks = run_all(rapido=True)
        assert sum(resumo(checks)) == len(checks)

    def test_sem_erros_sugere_execucao(self):
        checks = [Check("a", OK), Check("b", AVISO, solucao="instale x")]
        texto = proximo_passo(checks)
        assert "lprts demo" in texto

    def test_com_erro_manda_resolver_primeiro(self):
        checks = [Check("a", ERRO, "faltou", "pip install algo")]
        texto = proximo_passo(checks)
        assert "Resolva os erros" in texto and "pip install algo" in texto

    def test_solucoes_preservam_os_extras_do_pip(self):
        """Regressao: o rich removia '[dl]' interpretando como marcacao."""
        pendencias = [c for c in run_all(rapido=True) if c.solucao and "pip install" in c.solucao]
        for c in pendencias:
            if "-e" in c.solucao and "[" in c.solucao:
                assert "]" in c.solucao      # o extra chegou inteiro

    def test_ambiente_reportado(self):
        env = doctor.ambiente()
        assert "Sistema" in env and "Python" in env


class TestEstadoRapido:
    def test_modo_rapido_pula_o_teste_de_fumaca(self):
        nomes_rapido = {c.nome for c in run_all(rapido=True)}
        nomes_completo = {c.nome for c in run_all(rapido=False)}
        assert "Pipeline (teste de fumaca)" in nomes_completo
        assert "Pipeline (teste de fumaca)" not in nomes_rapido
