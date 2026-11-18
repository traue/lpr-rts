"""Taxonomia de objetos: categorias, porte de placa e presets."""

from __future__ import annotations

import pytest

from lprts.core import taxonomy as tx
from lprts.core.taxonomy import ObjectCategory, ObjectClass


class TestConversao:
    @pytest.mark.parametrize(
        "rotulo,esperado",
        [
            ("person", ObjectClass.PESSOA),
            ("bicycle", ObjectClass.BICICLETA),
            ("car", ObjectClass.CARRO),
            ("motorcycle", ObjectClass.MOTOCICLETA),
            ("truck", ObjectClass.CAMINHAO),
            ("dog", ObjectClass.CACHORRO),
        ],
    )
    def test_de_rotulo_coco(self, rotulo, esperado):
        assert tx.from_label(rotulo) is esperado

    def test_de_nome_interno(self):
        assert tx.from_label("carro") is ObjectClass.CARRO

    def test_de_id_coco(self):
        assert tx.from_coco_id(0) is ObjectClass.PESSOA
        assert tx.from_coco_id(2) is ObjectClass.CARRO

    def test_desconhecido_nao_levanta(self):
        assert tx.from_label("girafa_voadora") is ObjectClass.DESCONHECIDO
        assert tx.from_coco_id(999) is ObjectClass.DESCONHECIDO


class TestPorteDePlaca:
    @pytest.mark.parametrize(
        "cls", [ObjectClass.CARRO, ObjectClass.MOTOCICLETA, ObjectClass.ONIBUS, ObjectClass.CAMINHAO]
    )
    def test_veiculos_motorizados_tem_placa(self, cls):
        assert tx.has_plate(cls)

    @pytest.mark.parametrize(
        "cls", [ObjectClass.PESSOA, ObjectClass.BICICLETA, ObjectClass.CACHORRO, ObjectClass.SEMAFORO]
    )
    def test_demais_classes_nao_tem_placa(self, cls):
        assert not tx.has_plate(cls)

    def test_bicicleta_nao_e_emplacada_no_brasil(self):
        """Regra de dominio explicita: bicicleta nunca deve chegar ao OCR."""
        assert not tx.has_plate(ObjectClass.BICICLETA)
        assert tx.info(ObjectClass.BICICLETA).is_vehicle    # e veiculo, mas sem placa

    def test_conjunto_com_placa(self):
        assert tx.plate_bearing_classes() == {
            ObjectClass.CARRO, ObjectClass.MOTOCICLETA,
            ObjectClass.ONIBUS, ObjectClass.CAMINHAO,
        }


class TestVulneraveis:
    @pytest.mark.parametrize(
        "cls", [ObjectClass.PESSOA, ObjectClass.BICICLETA, ObjectClass.MOTOCICLETA]
    )
    def test_usuarios_vulneraveis(self, cls):
        assert tx.is_vulnerable(cls)

    def test_motocicleta_e_vulneravel_e_emplacada(self):
        """Caso unico: e o unico veiculo ao mesmo tempo vulneravel e com placa."""
        assert tx.is_vulnerable(ObjectClass.MOTOCICLETA)
        assert tx.has_plate(ObjectClass.MOTOCICLETA)

    @pytest.mark.parametrize("cls", [ObjectClass.CARRO, ObjectClass.ONIBUS, ObjectClass.CAMINHAO])
    def test_veiculos_fechados_nao_sao_vulneraveis(self, cls):
        assert not tx.is_vulnerable(cls)


class TestCategorias:
    def test_categoria_de_cada_classe(self):
        assert tx.category(ObjectClass.PESSOA) is ObjectCategory.PESSOA
        assert tx.category(ObjectClass.CARRO) is ObjectCategory.VEICULO_MOTORIZADO
        assert tx.category(ObjectClass.BICICLETA) is ObjectCategory.VEICULO_NAO_MOTORIZADO
        assert tx.category(ObjectClass.CACHORRO) is ObjectCategory.ANIMAL
        assert tx.category(ObjectClass.SEMAFORO) is ObjectCategory.MOBILIARIO_URBANO

    def test_classes_por_categoria(self):
        motorizados = tx.classes_in(ObjectCategory.VEICULO_MOTORIZADO)
        assert ObjectClass.CARRO in motorizados
        assert ObjectClass.BICICLETA not in motorizados


class TestPresets:
    def test_preset_veiculos_so_tem_placa(self):
        ids = set(tx.preset("veiculos"))
        assert ids == {tx.info(c).coco_id for c in tx.plate_bearing_classes()}

    def test_preset_trafego_inclui_pessoas_e_bicicletas(self):
        nomes = set(tx.preset("trafego").values())
        assert {"person", "bicycle", "car", "motorcycle"} <= nomes

    def test_presets_crescem_em_abrangencia(self):
        assert len(tx.preset("veiculos")) < len(tx.preset("trafego")) < len(tx.preset("completo"))

    def test_preset_invalido_levanta(self):
        with pytest.raises(ValueError):
            tx.preset("inexistente")


class TestApresentacao:
    def test_cor_por_classe(self):
        cor = tx.color(ObjectClass.CARRO)
        assert len(cor) == 3 and all(0 <= c <= 255 for c in cor)

    def test_classes_tem_cores_distintas(self):
        """Cores repetidas tornariam a interface ambigua."""
        principais = [
            ObjectClass.PESSOA, ObjectClass.BICICLETA, ObjectClass.CARRO,
            ObjectClass.MOTOCICLETA, ObjectClass.ONIBUS, ObjectClass.CAMINHAO,
        ]
        assert len({tx.color(c) for c in principais}) == len(principais)

    def test_rotulo_em_portugues(self):
        assert tx.label_pt(ObjectClass.MOTOCICLETA) == "motocicleta"
