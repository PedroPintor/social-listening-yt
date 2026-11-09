"""Testes do pipeline com a API simulada (sem rede, sem cota):  python teste_mock.py"""

import json
import tempfile
from pathlib import Path

import gerar_amostra as ga
from mock_api import criar_transporte
from youtube import YouTube

D, A = "2026-11-02T00:00:00Z", "2026-11-09T00:00:00Z"


def teste_cliente():
    with tempfile.TemporaryDirectory() as tmp:
        yt = YouTube(transporte=criar_transporte(), pasta_cache=tmp)
        vids = yt.buscar_videos("iPhone 17", D, A, max_videos=12)
        assert len(vids) == 12 and yt.unidades == 100
        det = yt.detalhes_videos([v["video_id"] for v in vids])
        assert len(det) == 11 and not any(v["infantil"] for v in det), "madeForKids deve ser pulado"
        assert yt.unidades == 101
        assert yt.comentarios("iPho05") == [], "commentsDisabled deve virar lista vazia"
        cs = yt.comentarios("iPho01", limite=150)
        assert len(cs) == 150 and yt.unidades == 104  # 1 (desativado) + 2 páginas
        cache_txt = " ".join(p.read_text(encoding="utf-8") for p in Path(tmp).rglob("*.json"))
        assert "author" not in cache_txt and "pessoa_real" not in cache_txt, "autor não pode ir para o cache"
        assert all(set(c) == {"video_id", "comment_id", "texto", "likes", "data_comentario", "n_respostas"} for c in cs)

        yt2 = YouTube(transporte=criar_transporte(), pasta_cache=tmp)  # nova execução, mesmo cache
        assert yt2.comentarios("iPho01", limite=150) == cs and yt2.unidades == 0 and yt2.acertos_cache == 2


def teste_cota():
    yt = YouTube(transporte=criar_transporte(quota_apos=3), pasta_cache=None)
    videos, coments = ga.coletar(yt, D, A)
    assert yt.cota_esgotada and len(coments) > 0, "deve parar sem quebrar e manter o que já coletou"
    assert yt.comentarios("xx01") == [] and yt.buscar_videos("x", D, A) == []


def teste_filtros():
    assert ga.eh_portugues("Não gostei, o som é muito baixo")
    assert ga.eh_portugues("mto bom kkkk recomendo demais")
    assert not ga.eh_portugues("Este producto es muy bueno pero caro")
    assert not ga.eh_portugues("I love this phone so much")
    assert not ga.comentario_valido("Primeiro!")
    assert not ga.comentario_valido("Ganhe dinheiro em casa acesse https://x.com agora")
    assert not ga.comentario_valido("Se inscreve no meu canal que eu sigo")
    assert ga.comentario_valido("A bateria do celular acaba rápido demais")


def teste_amostra():
    yt = YouTube(transporte=criar_transporte(), pasta_cache=None)
    videos, coments = ga.coletar(yt, D, A)
    am1, am2 = ga.amostrar(videos, coments), ga.amostrar(videos, coments)
    assert am1.equals(am2), "semente fixa: amostra reprodutível"
    assert list(am1.columns) == ["id", "tipo", "produto", "video_id", "comment_id", "titulo_video", "data_video", "texto", "likes"]
    assert am1["id"].is_unique
    c = am1[am1["tipo"] == "comentário"]
    assert c.groupby("video_id").size().max() <= ga.MAX_POR_VIDEO
    assert c.groupby("produto").size().max() <= ga.AMOSTRA_POR_PRODUTO
    assert (am1["tipo"] == "título").sum() == ga.TITULOS_TOTAL
    assert not c["texto"].str.contains("http").any()
    assert set(am1["produto"]) == {p["produto"] for p in ga.PRODUTOS}
    json.dumps(am1.to_dict("records"), ensure_ascii=False)


if __name__ == "__main__":
    for f in (teste_cliente, teste_cota, teste_filtros, teste_amostra):
        f()
        print("ok", f.__name__)
    print("Todos os testes passaram (nenhuma chamada real à API).")
