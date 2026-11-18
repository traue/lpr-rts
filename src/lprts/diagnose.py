"""Diagnostico de pipeline: mostra em que estagio o fluxo para.

Existe por um motivo pratico. Quando a tabela de placas fica vazia, a causa pode
estar em qualquer um de seis pontos, e a mensagem de erro nao existe - o sistema
simplesmente nao produz nada. Sem instrumentacao, restam palpites.

Este modulo roda alguns quadros contando o que sai de cada estagio::

    quadros -> objetos -> objetos com placa -> placas -> trilhas -> leituras -> eventos

O primeiro contador que zera e a causa. O relatorio nomeia esse estagio e o que
fazer a respeito, em vez de listar tudo o que poderia estar errado.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .core.taxonomy import has_plate
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class StageCounts:
    """Contagens por estagio ao longo dos quadros analisados."""

    frames: int = 0
    objetos: int = 0
    objetos_com_placa: int = 0
    placas: int = 0
    trilhas: int = 0
    trilhas_confirmadas: int = 0
    leituras: int = 0
    leituras_validas: int = 0
    eventos: int = 0

    classes_vistas: Counter = field(default_factory=Counter)
    erros_ocr: set[str] = field(default_factory=set)
    larguras_de_placa: list[float] = field(default_factory=list)
    fallback_ativo: bool = False
    fallback_motivo: str = ""
    modo_so_placa: bool = False
    detector_objetos: str = ""
    detector_placas: str = ""
    motor_ocr: str = ""
    dispositivo: str = ""

    @property
    def largura_mediana(self) -> float:
        if not self.larguras_de_placa:
            return 0.0
        ordenado = sorted(self.larguras_de_placa)
        return ordenado[len(ordenado) // 2]

    @property
    def largura_maxima(self) -> float:
        return max(self.larguras_de_placa, default=0.0)


@dataclass(slots=True)
class Diagnosis:
    """Contagens mais o veredito."""

    counts: StageCounts
    estagio_quebrado: str = ""
    causa: str = ""
    solucao: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.estagio_quebrado


def _analisar(c: StageCounts, cfg: AppConfig) -> Diagnosis:
    """Encontra o estagio que bloqueia o fluxo e explica o porque.

    O criterio e: o primeiro estagio que **recebeu entrada e nao produziu
    saida**. Nao basta procurar o primeiro contador zerado - um estagio pode
    zerar legitimamente e o fluxo seguir por outro caminho. E o que acontece com
    o fallback automatico: a deteccao de objetos fica em zero, o rastreio de
    placas assume, e apontar "deteccao de objetos" mandaria investigar um
    estagio que foi contornado com sucesso.
    """
    d = Diagnosis(counts=c)

    # Se placas foram publicadas, o pipeline funcionou de ponta a ponta.
    if c.eventos > 0:
        return d

    if c.frames == 0:
        d.estagio_quebrado = "captura"
        d.causa = "Nenhum quadro foi lido da fonte."
        d.solucao = [
            "Confira o caminho do arquivo ou o indice da camera.",
            "No macOS, a webcam exige permissao: Ajustes > Privacidade e Seguranca > Camera.",
        ]
        return d

    # Os estagios de objeto sao contornados quando as placas sao rastreadas
    # diretamente - seja por configuracao, seja pelo fallback em tempo de execucao.
    objetos_irrelevantes = c.modo_so_placa or c.fallback_ativo

    if not objetos_irrelevantes:
        if c.objetos == 0:
            d.estagio_quebrado = "deteccao de objetos"
            d.causa = (
                "O detector de objetos nao encontrou nada em nenhum quadro. "
                "Isso e esperado em video sintetico, animacao ou imagem termica: "
                "ele foi treinado em fotos reais."
            )
            d.solucao = [
                "Rode com --no-vehicle-detector para rastrear as placas diretamente.",
                "Ou baixe o limiar: --config com detector.conf menor (padrao 0.30).",
            ]
            return d

        if c.objetos_com_placa == 0:
            vistas = ", ".join(f"{k} ({v})" for k, v in c.classes_vistas.most_common(5))
            d.estagio_quebrado = "objetos que podem portar placa"
            d.causa = (
                f"O detector encontrou objetos, mas nenhum que porte placa. Viu: {vistas}. "
                "Placas so sao procuradas dentro de carros, motos, onibus e caminhoes - "
                "pedestres e bicicletas sao pulados de proposito."
            )
            d.solucao = [
                "Rode com --no-vehicle-detector para procurar placas no quadro inteiro.",
                "Se o video tem veiculos reais, o detector pode estar com limiar alto: "
                "reduza detector.conf.",
            ]
            return d

    if c.placas == 0:
        d.estagio_quebrado = "deteccao de placas"
        d.causa = (
            f"Nenhuma placa foi localizada em {c.frames} quadros "
            f"(detector: {c.detector_placas})."
        )
        d.solucao = [
            "Baixe o limiar do detector de placas: detector.plate_conf (padrao 0.25).",
            "O baseline classico precisa de placas nitidas e frontais; "
            "considere treinar um detector (scripts/train_plate_detector.py).",
        ]
        return d

    if c.trilhas == 0:
        d.estagio_quebrado = "rastreamento"
        d.causa = (
            f"{c.placas} placa(s) detectada(s), mas nenhuma trilha foi criada."
        )
        d.solucao = [
            "Reduza tracker.min_hits (padrao 3) - em video curto pode nao dar tempo.",
            "Reduza tracker.plate_high_thresh se o detector devolve confianca baixa.",
        ]
        return d

    if c.leituras == 0:
        largura = c.largura_mediana
        if largura and largura < cfg.ocr.min_plate_width:
            d.estagio_quebrado = "OCR (placa pequena demais)"
            d.causa = (
                f"As placas detectadas tem mediana de {largura:.0f} px de largura, "
                f"abaixo do minimo de {cfg.ocr.min_plate_width} px para tentar OCR. "
                f"A maior vista tinha {c.largura_maxima:.0f} px."
            )
            d.solucao = [
                f"Reduza ocr.min_plate_width para ~{max(30, int(largura * 0.8))}.",
                "Ou aproxime a camera / aumente a resolucao da fonte.",
            ]
            return d

        d.estagio_quebrado = f"OCR (motor: {c.motor_ocr})"
        d.causa = (
            f"{c.trilhas_confirmadas} trilha(s) confirmada(s) e placas de tamanho "
            f"suficiente (mediana {largura:.0f} px), mas o motor '{c.motor_ocr}' nao "
            "devolveu nenhuma leitura. Quase sempre isso significa que o motor esta "
            "falhando a cada chamada - e a falha estava sendo engolida."
        )
        d.solucao = [
            "Rode de novo com --log-level DEBUG para ver o erro do motor.",
            "Troque de motor para isolar o problema: --ocr template "
            "(baseline interno, sempre funciona).",
        ]
        if c.erros_ocr:
            exemplo = next(iter(c.erros_ocr))
            d.causa += f" Erro observado: {exemplo}"
            d.solucao.insert(0, "Veja o erro relatado acima.")
        return d

    if c.eventos == 0:
        taxa = c.leituras_validas / c.leituras if c.leituras else 0.0
        d.estagio_quebrado = "fusao (publicacao)"
        d.causa = (
            f"{c.leituras} leitura(s) de OCR, mas nenhuma foi publicada. "
            f"Apenas {taxa:.0%} fecharam a gramatica brasileira, e a fusao exige "
            f"{cfg.fusion.min_reads} leituras concordantes."
        )
        d.solucao = [
            f"Reduza fusion.min_reads (atual {cfg.fusion.min_reads}) para 2.",
            f"Reduza fusion.min_agreement (atual {cfg.fusion.min_agreement}).",
        ]
        if taxa < 0.3:
            d.solucao.append(
                "A taxa de sintaxe valida esta baixa: o OCR esta errando. "
                'Instale um motor profundo: pip install -e ".[ocr]"'
            )
        return d

    return d


def run_diagnosis(
    config: AppConfig, max_frames: int = 150, progress=None
) -> Diagnosis:
    """Executa o pipeline por ``max_frames`` quadros, contando cada estagio."""
    from .factory import build_pipeline, build_source

    cfg = AppConfig.from_dict(config.to_dict())     # copia: nao altera a original
    cfg.source.max_frames = max_frames
    cfg.source.realtime = False
    cfg.analytics.save_crops = False

    c = StageCounts()
    source = build_source(cfg)
    pipeline = build_pipeline(cfg, store=None, warmup=True)

    c.modo_so_placa = pipeline._plate_only_mode
    c.detector_objetos = (
        pipeline.vehicle_detector.name if pipeline.vehicle_detector else "desabilitado"
    )
    c.detector_placas = pipeline.plate_detector.name if pipeline.plate_detector else "nenhum"
    c.motor_ocr = pipeline.ocr_engine.name if pipeline.ocr_engine else "nenhum"
    c.dispositivo = pipeline.device_summary()

    try:
        for resultado in pipeline.run(source):
            c.frames += 1
            c.objetos += len(resultado.vehicles)
            for v in resultado.vehicles:
                c.classes_vistas[v.label] += 1
                if has_plate(v.label):
                    c.objetos_com_placa += 1
            c.placas += len(resultado.plates)
            for p in resultado.plates:
                c.larguras_de_placa.append(p.bbox[2] - p.bbox[0])
            c.trilhas += len(resultado.tracks)
            c.trilhas_confirmadas += sum(1 for t in resultado.tracks if t.confirmed)
            c.leituras += len(resultado.reads)
            c.leituras_validas += sum(1 for r in resultado.reads if r.valid_syntax)
            if progress is not None:
                progress(c.frames)
        pipeline.finalize()
        c.eventos = pipeline.metrics.snapshot().n_events
        c.fallback_ativo = pipeline.fallback_active
        c.fallback_motivo = pipeline.fallback_reason
    finally:
        pipeline.stop()
        source.close()

    return _analisar(c, cfg)


def relatorio(d: Diagnosis, fonte: str | Path = "") -> str:
    """Relatorio em texto puro do diagnostico."""
    c = d.counts
    linhas = [
        "=" * 74,
        "DIAGNOSTICO DO PIPELINE",
        "=" * 74,
        f"Fonte .............: {fonte}",
        f"Quadros analisados : {c.frames}",
        f"Detector de objetos: {c.detector_objetos}",
        f"Detector de placas : {c.detector_placas}",
        f"Motor de OCR ......: {c.motor_ocr}",
        f"Dispositivo .......: {c.dispositivo}",
        f"Modo so-placa .....: {'sim' if c.modo_so_placa else 'nao'}",
    ]
    if c.fallback_ativo:
        linhas.append(f"Fallback acionado .: sim - {c.fallback_motivo}")

    linhas += ["", "-" * 74, "FLUXO POR ESTAGIO (soma sobre todos os quadros)", "-" * 74]
    # No modo so-placa os estagios de objeto sao inaplicaveis por construcao:
    # marca-los como falha apontaria para o lugar errado.
    na = c.modo_so_placa or c.fallback_ativo
    estagios = [
        ("quadros lidos", c.frames, False),
        ("objetos detectados", c.objetos, na),
        ("...que podem portar placa", c.objetos_com_placa, na),
        ("placas detectadas", c.placas, False),
        ("trilhas ativas", c.trilhas, False),
        ("...confirmadas", c.trilhas_confirmadas, False),
        ("leituras de OCR", c.leituras, False),
        ("...com sintaxe valida", c.leituras_validas, False),
        ("EVENTOS PUBLICADOS", c.eventos, False),
    ]
    for nome, valor, inaplicavel in estagios:
        if inaplicavel:
            linhas.append(f"  - {nome:.<40} (nao se aplica no modo so-placa)")
        else:
            marca = "  " if valor > 0 else " X"
            linhas.append(f"{marca} {nome:.<40} {valor}")

    if c.classes_vistas:
        top = ", ".join(f"{k}={v}" for k, v in c.classes_vistas.most_common(6))
        linhas += ["", f"  Classes vistas: {top}"]
    if c.larguras_de_placa:
        linhas.append(
            f"  Largura das placas: mediana {c.largura_mediana:.0f} px, "
            f"maxima {c.largura_maxima:.0f} px"
        )

    linhas += ["", "-" * 74]
    if d.ok:
        linhas += [
            "RESULTADO: o pipeline funciona de ponta a ponta.",
            "-" * 74,
            f"  {c.eventos} placa(s) publicada(s) em {c.frames} quadros.",
        ]
        if c.fallback_ativo:
            linhas += [
                "",
                "  Observacao: o detector de objetos nao serviu para este video e o",
                "  sistema passou a rastrear as placas diretamente. Funciona, mas gasta",
                "  os primeiros quadros a toa. Para evitar isso, rode com:",
                "    --no-vehicle-detector --plate-detector classic",
            ]
    else:
        linhas += [
            f"O FLUXO PARA EM: {d.estagio_quebrado}",
            "-" * 74,
            f"  {d.causa}",
            "",
            "  O que fazer:",
        ]
        linhas += [f"    - {s}" for s in d.solucao]
    linhas.append("=" * 74)
    return "\n".join(linhas)
