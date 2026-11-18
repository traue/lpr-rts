"""Configuracao de runtimes nativos antes de qualquer sessao de inferencia.

Existe por causa de uma falha concreta, diagnosticada por relatorio de crash do
macOS: o processo abortava ao encerrar, com esta pilha::

    Microsoft::Applications::Events::HttpClientManager::onHttpResponse
    Microsoft::Applications::Events::LogManagerImpl::DispatchEvent
    onnxruntime::PosixTelemetry::Shutdown()
    __cxa_finalize_ranges -> exit -> Py_Exit

O ONNX Runtime embute o SDK de telemetria da Microsoft e, no encerramento, tenta
enviar dados de uso por HTTP. A thread de trabalho da telemetria processa a
resposta enquanto a thread principal ja destruiu o ``recursive_mutex`` que ela
tenta travar - excecao de C++ nao capturada, ``abort()``.

Nao e um defeito do pipeline: e o runtime "telefonando para casa" durante a
propria destruicao. Desligar a telemetria elimina a corrida na origem, e num
projeto de pesquisa e o padrao correto de qualquer forma - nenhum dado de uso
deveria sair da maquina sem que o pesquisador saiba.
"""

from __future__ import annotations

import os

from .utils.logging import get_logger

log = get_logger(__name__)

_CONFIGURADO = False


def configure_onnxruntime(force: bool = False) -> bool:
    """Desliga a telemetria do ONNX Runtime. Idempotente e seguro sem a lib.

    Precisa rodar **antes** da criacao de qualquer sessao. Devolve True se a
    configuracao foi aplicada.
    """
    global _CONFIGURADO
    if _CONFIGURADO and not force:
        return True

    # A variavel de ambiente cobre subprocessos e bibliotecas que criam sessoes
    # por conta propria (o fast-plate-ocr, por exemplo) antes de chegarmos nelas.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

    try:
        import onnxruntime as ort
    except ImportError:
        _CONFIGURADO = True      # sem ORT nao ha o que configurar
        return False

    try:
        ort.disable_telemetry_events()
        log.debug("Telemetria do ONNX Runtime desligada.")
    except Exception as exc:  # noqa: BLE001 - versoes antigas podem nao expor
        log.debug("Nao foi possivel desligar a telemetria do ONNX Runtime: %s", exc)

    try:
        # 3 = ERROR. O padrao inunda o terminal com avisos de particionamento
        # do CoreML a cada sessao criada.
        ort.set_default_logger_severity(3)
    except Exception as exc:  # noqa: BLE001
        log.debug("Nao foi possivel ajustar o nivel de log do ONNX Runtime: %s", exc)

    _CONFIGURADO = True
    return True


_VARS_THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_native_threads(n_workers: int) -> int:
    """Reparte os nucleos quando N processos rodam o pipeline ao mesmo tempo.

    Cada biblioteca nativa envolvida - OpenCV, PyTorch, ONNX Runtime, BLAS -
    dimensiona o proprio conjunto de threads pelo numero de nucleos da maquina.
    Com N processos simultaneos isso significa N vezes mais threads que nucleos,
    e os workers passam a disputar os mesmos recursos.

    As variaveis de ambiente so tem efeito se forem definidas **antes** de a
    biblioteca ser importada - por isso o pool de processos usa o metodo
    ``spawn`` e chama esta funcao no inicializador de cada worker. Devolve
    quantas threads ficaram para cada processo.

    O que isto resolve e o que nao resolve, medido com 6 videos de 400 quadros
    numa maquina de 4 nucleos:

    - Com detector classico e OCR por gabaritos (sem thread pool nativo
      proprio), o lote em 3 processos leva 9,6 s contra 23,1 s em serie - 2,4x.
    - Com a configuracao padrao (YOLO em PyTorch + OCR em ONNX Runtime), o lote
      em 3 processos leva 209 s contra 74 s em serie. Limitar as threads **nao**
      recupera esse caso: cada worker ja satura a maquina sozinho, porque o
      PyTorch paraleliza a inferencia internamente. Nessa configuracao o ganho
      de rodar varios videos ao mesmo tempo nao existe numa unica maquina com
      poucos nucleos - e para isso que serve o backend MPI, com um video por no.
    """
    total = os.cpu_count() or 4
    por_worker = max(1, total // max(1, n_workers))
    if n_workers <= 1:
        return total

    for var in _VARS_THREADS:
        os.environ[var] = str(por_worker)

    try:
        import cv2

        cv2.setNumThreads(por_worker)
    except Exception as exc:  # noqa: BLE001 - OpenCV pode nao estar carregado
        log.debug("Nao foi possivel limitar as threads do OpenCV: %s", exc)

    try:
        import torch

        torch.set_num_threads(por_worker)
    except Exception as exc:  # noqa: BLE001 - torch e opcional
        log.debug("Nao foi possivel limitar as threads do PyTorch: %s", exc)

    log.debug("Threads nativas limitadas a %d por worker (%d workers).", por_worker, n_workers)
    return por_worker


def import_ultralytics():
    """Importa o ultralytics sem deixar o OpenCV do processo numa thread so.

    Ao ser importado, o ultralytics chama ``cv2.setNumThreads(0)`` - pensado
    para o DataLoader de treino - e todo o OpenCV do processo passa a rodar
    numa thread. Medido num video 4K vertical de iPhone: redimensionar o quadro
    para 1280 px (fator nao inteiro, ``INTER_AREA``) passou de ~7 ms para 70 ms
    e o processamento caiu para ~10 quadros/s so por ligar o YOLO. Aqui o
    numero de threads de antes do import (o padrao, ou o limite por processo do
    lote) e restaurado. O script de treino importa o ultralytics direto e
    mantem o comportamento dele.
    """
    import cv2

    antes = cv2.getNumThreads()
    import ultralytics

    if cv2.getNumThreads() != antes:
        cv2.setNumThreads(antes)
        log.debug("Threads do OpenCV restauradas para %d apos importar o ultralytics.", antes)
    return ultralytics


def configure_all() -> None:
    """Ponto unico chamado no arranque, antes de montar qualquer componente."""
    configure_onnxruntime()
