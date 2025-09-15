#!/usr/bin/env python3
"""
Ejemplo de uso de búsqueda semántica con Pinecone.

Este script demuestra cómo usar la clase PineconeSemanticSearch
para realizar búsquedas semánticas en los datos migrados desde BigQuery.
"""

import logging
import os
import sys

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from pinecone_search import PineconeSemanticSearch

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def example_company_profiles_search():
    """Ejemplo de búsqueda en perfiles de empresas."""
    print("\n" + "=" * 60)
    print("BÚSQUEDA EN PERFILES DE EMPRESAS")
    print("=" * 60)

    try:
        searcher = PineconeSemanticSearch()

        # Búsqueda general
        print("\n1. Búsqueda general: 'tecnología software innovación'")
        results = searcher.search_company_profiles(query="tecnología software innovación", limit=5)

        if not results.empty:
            print(f"Encontradas {len(results)} empresas:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']}: {row['companyName']} ({row['sector']}) - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda con filtros
        print("\n2. Búsqueda con filtros: 'banca fintech' en sector Technology")
        results = searcher.search_company_profiles(
            query="banca fintech pagos digitales", limit=3, sector_filter="Technology"
        )

        if not results.empty:
            print(f"Encontradas {len(results)} empresas:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']}: {row['companyName']} ({row['industry']}) - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda por país
        print("\n3. Búsqueda por país: 'automóviles' en España")
        results = searcher.search_company_profiles(
            query="automóviles vehículos transporte", limit=3, country_filter="Spain"
        )

        if not results.empty:
            print(f"Encontradas {len(results)} empresas:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']}: {row['companyName']} ({row['exchange']}) - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

    except Exception as e:
        logger.error(f"Error en búsqueda de perfiles: {e}")


def example_earnings_transcripts_search():
    """Ejemplo de búsqueda en transcripciones de ganancias."""
    print("\n" + "=" * 60)
    print("BÚSQUEDA EN TRANSCRIPCIONES DE GANANCIAS")
    print("=" * 60)

    try:
        searcher = PineconeSemanticSearch()

        # Búsqueda general
        print("\n1. Búsqueda general: 'crecimiento ingresos'")
        results = searcher.search_earnings_transcripts(query="crecimiento ingresos rentabilidad", limit=5)

        if not results.empty:
            print(f"Encontradas {len(results)} transcripciones:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']} Q{row['quarter']} {row['year']} - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda por empresa específica
        print("\n2. Búsqueda por empresa: 'Apple'")
        results = searcher.search_earnings_transcripts(query="iPhone ventas servicios", limit=3, symbol_filter="AAPL")

        if not results.empty:
            print(f"Encontradas {len(results)} transcripciones de Apple:")
            for _, row in results.iterrows():
                print(f"  - Q{row['quarter']} {row['year']} - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda por año
        print("\n3. Búsqueda por año: '2024'")
        results = searcher.search_earnings_transcripts(
            query="inteligencia artificial machine learning", limit=3, year_filter=2024
        )

        if not results.empty:
            print(f"Encontradas {len(results)} transcripciones de 2024:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']} Q{row['quarter']} - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

    except Exception as e:
        logger.error(f"Error en búsqueda de transcripciones: {e}")


def example_news_search():
    """Ejemplo de búsqueda en noticias financieras."""
    print("\n" + "=" * 60)
    print("BÚSQUEDA EN NOTICIAS FINANCIERAS")
    print("=" * 60)

    try:
        searcher = PineconeSemanticSearch()

        # Búsqueda general
        print("\n1. Búsqueda general: 'mercado acciones'")
        results = searcher.search_news(query="mercado acciones volatilidad trading", limit=5)

        if not results.empty:
            print(f"Encontradas {len(results)} noticias:")
            for _, row in results.iterrows():
                print(f"  - {row['title'][:80]}... - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda por tipo de noticia
        print("\n2. Búsqueda en noticias de empresa: 'Tesla'")
        results = searcher.search_news(
            query="Tesla vehículos eléctricos autopilot", limit=3, news_type_filter="company-news"
        )

        if not results.empty:
            print(f"Encontradas {len(results)} noticias de empresa:")
            for _, row in results.iterrows():
                print(f"  - {row['title'][:80]}... - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

        # Búsqueda con filtro de fecha
        print("\n3. Búsqueda reciente: 'crypto bitcoin'")
        results = searcher.search_news(query="crypto bitcoin blockchain", limit=3, date_from_filter="2024-01-01")

        if not results.empty:
            print(f"Encontradas {len(results)} noticias recientes:")
            for _, row in results.iterrows():
                print(f"  - {row['title'][:80]}... ({row['publish_date']}) - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

    except Exception as e:
        logger.error(f"Error en búsqueda de noticias: {e}")


def example_custom_search():
    """Ejemplo de búsqueda personalizada."""
    print("\n" + "=" * 60)
    print("BÚSQUEDA PERSONALIZADA")
    print("=" * 60)

    try:
        searcher = PineconeSemanticSearch()

        # Búsqueda con configuración personalizada
        print("\n1. Búsqueda con umbral alto: 'energía renovable'")
        results = searcher.search(
            query="energía renovable solar eólica",
            index_name="profiles-df",
            limit=10,
            threshold=0.8,  # Solo resultados muy similares
            filter_dict={"sector": "Energy"},
        )

        if not results.empty:
            print(f"Encontradas {len(results)} empresas de energía:")
            for _, row in results.iterrows():
                print(f"  - {row['symbol']}: {row['companyName']} - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados con umbral alto")

        # Búsqueda sin metadatos
        print("\n2. Búsqueda sin metadatos: 'farmacéutica'")
        results = searcher.search(
            query="farmacéutica medicamentos investigación", index_name="profiles-df", limit=5, include_metadata=False
        )

        if not results.empty:
            print(f"Encontradas {len(results)} resultados:")
            for _, row in results.iterrows():
                print(f"  - ID: {row['id']} - Score: {row['score']:.3f}")
        else:
            print("No se encontraron resultados")

    except Exception as e:
        logger.error(f"Error en búsqueda personalizada: {e}")


def example_index_stats():
    """Ejemplo de obtención de estadísticas de índices."""
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE ÍNDICES")
    print("=" * 60)

    try:
        searcher = PineconeSemanticSearch()

        # Obtener estadísticas de diferentes índices
        indices = ["profiles-df", "earnings-transcripts", "investpy-news"]

        for index_name in indices:
            try:
                stats = searcher.get_index_stats(index_name)
                print(f"\n{index_name}:")
                print(f"  - Total vectores: {stats['total_vector_count']:,}")
                print(f"  - Dimensión: {stats['dimension']}")
                print(f"  - Completitud: {stats['index_fullness']:.2%}")
            except Exception as e:
                print(f"\n{index_name}: Error obteniendo estadísticas - {e}")

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("EJEMPLOS DE BÚSQUEDA SEMÁNTICA CON PINECONE")
    print("=" * 60)

    # Verificar configuración
    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY no está configurado")
        print("Configura la variable de entorno PINECONE_API_KEY")
        return

    try:
        # Ejecutar ejemplos
        example_company_profiles_search()
        example_earnings_transcripts_search()
        example_news_search()
        example_custom_search()
        example_index_stats()

        print("\n" + "=" * 60)
        print("TODOS LOS EJEMPLOS COMPLETADOS")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error en ejemplos: {e}")


if __name__ == "__main__":
    main()
