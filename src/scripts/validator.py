import os
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

print(os.getcwd())
# Add the correct paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/utils/ml/embeddings/src


from embeddings import SemanticSearch
from utils.utils import load_query_embeddings

load_dotenv()

queries_list = ["news_semantic_search.sql", "company_profiles_semantic_search.sql", "earnings_transcripts_search.sql"]


class EmbeddingValidator:
    """
    Class to validate and compare different embedding models using Streamlit.
    """

    def __init__(self):
        """
        Initializes the embedding validator.
        """
        self.semantic_search = SemanticSearch()
        self.available_models = ["text_embedding", "text_embedding_gemini"]

    def get_available_queries(self) -> Dict[str, str]:
        """
        Returns a dictionary with available SQL queries for testing.
        """
        queries = {}

        # Load queries from SQL files
        for query_file in queries_list:
            try:
                # Build the complete path to the query
                query_path = f"big_query/select_queries/{query_file}"
                query_content = load_query_embeddings(query_path)

                # Generate readable name for the query
                query_name = query_file.replace(".sql", "").replace("_", " ").title()
                queries[query_name] = query_content

            except Exception as e:
                st.warning(f"Could not load query {query_file}: {str(e)}")
                # Fallback: basic query
                queries[query_name] = f"-- Error loading {query_file}\n-- {str(e)}"

        return queries

    def validate_query_placeholders(self, query_content: str) -> Dict[str, bool]:
        """
        Validates that the query has the necessary placeholders.

        Args:
            query_content: SQL query content

        Returns:
            Dictionary with the validation status of each placeholder
        """
        required_placeholders = [
            "{user_query}",
            "{model}",
            "{project_id}",
            "{database}",
            "{table_name}",
            "{limit_results}",
        ]

        validation = {}
        for placeholder in required_placeholders:
            validation[placeholder] = placeholder in query_content

        return validation

    def display_embedding_metrics(self, results: Dict[str, pd.DataFrame]):
        """
        Displays typical embedding metrics comparing results across models.

        Args:
            results: Dictionary with results by model
        """
        if len(results) < 2:
            return

        # Create overlap metrics
        st.subheader("Result Overlap Analysis")

        # Identify unique results by title and date
        all_results = []
        for model, df in results.items():
            if not df.empty:
                for idx, row in df.iterrows():
                    identifier = row.get("title", "No Title")
                    all_results.append(
                        {
                            "model": model,
                            "identifier": identifier,
                            "title": row.get("title", "No Title"),
                            "distance": row.get("distance", 0),
                            "ranking": idx,  # DataFrame index
                        }
                    )

        if not all_results:
            st.info("No results to analyze for overlap.")
            return

        # Create analysis DataFrame
        analysis_df = pd.DataFrame(all_results)

        # Metrics by model
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Results per Model**")
            for model in results.keys():
                model_results = analysis_df[analysis_df["model"] == model]
                st.metric(f"{model}", len(model_results))

        with col2:
            st.markdown("**Unique Results**")
            unique_identifiers = analysis_df["identifier"].nunique()
            st.metric("Total unique", unique_identifiers)

            # Results that appear in multiple models
            overlap_results = analysis_df.groupby("identifier").filter(lambda x: len(x) > 1)
            st.metric("In multiple models", len(overlap_results))

        with col3:
            st.markdown("**Distance Statistics**")
            if "distance" in analysis_df.columns:
                avg_distance = analysis_df["distance"].mean()
                st.metric("Avg distance", f"{avg_distance:.4f}")

                min_distance = analysis_df["distance"].min()
                st.metric("Best distance", f"{min_distance:.4f}")

        # Overlap analysis
        st.subheader("Cross-Model Overlap")

        # Create overlap matrix
        overlap_matrix = []
        models_list = list(results.keys())

        for i, model1 in enumerate(models_list):
            row = []
            for j, model2 in enumerate(models_list):
                if i == j:
                    row.append(len(results[model1]) if not results[model1].empty else 0)
                else:
                    # Count results that appear in both models
                    if not results[model1].empty and not results[model2].empty:
                        model1_ids = set(row.get("title", "No Title") for _, row in results[model1].iterrows())
                        model2_ids = set(row.get("title", "No Title") for _, row in results[model2].iterrows())
                        overlap_count = len(model1_ids.intersection(model2_ids))
                        row.append(overlap_count)
                    else:
                        row.append(0)
            overlap_matrix.append(row)

        # Show overlap matrix
        overlap_df = pd.DataFrame(overlap_matrix, columns=models_list, index=models_list)
        st.markdown("**Overlap Matrix (number of shared results)**")
        st.dataframe(overlap_df, use_container_width=True)

        # Show results that appear in multiple models
        if len(overlap_results) > 0:
            st.subheader("Results Found in Multiple Models")

            # Group by identifier and calculate sum of positions
            grouped_overlap = overlap_results.groupby("identifier")

            # Create list with sum of positions to sort
            overlap_with_sum = []
            for identifier, group in grouped_overlap:
                # Calculate sum of real rankings (position + 1)
                total_ranking = sum(row["ranking"] + 1 for _, row in group.iterrows())
                overlap_with_sum.append(
                    {
                        "identifier": identifier,
                        "group": group,
                        "total_ranking": total_ranking,
                        "title": group.iloc[0]["title"],
                    }
                )

            # Sort by sum of positions (lower sum = better combined ranking)
            overlap_with_sum.sort(key=lambda x: x["total_ranking"])

            for item in overlap_with_sum[:10]:  # Show only the first 10
                group = item["group"]
                with st.expander(f"📰 {item['title'][:100]}... (Total Ranking: {item['total_ranking']})"):
                    # Show ranking in each model
                    for _, row in group.iterrows():
                        st.markdown(
                            f"- **{row['model']}**: Ranking #{row['ranking'] + 1}, Distance: {row['distance']:.4f}"
                        )

                    # Show description if available
                    if "description" in results[group.iloc[0]["model"]].columns:
                        desc_row = results[group.iloc[0]["model"]].iloc[group.iloc[0]["ranking"]]
                        if "description" in desc_row and pd.notna(desc_row["description"]):
                            st.markdown("**Description:**")
                            st.markdown(
                                f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                {str(desc_row["description"])}
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Show content if available
                    if "content" in results[group.iloc[0]["model"]].columns:
                        content_row = results[group.iloc[0]["model"]].iloc[group.iloc[0]["ranking"]]
                        if "content" in content_row:
                            st.markdown("**Content:**")
                            st.markdown(
                                f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                {str(content_row["content"])}
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Show unique results by model
        if len(results) == 2:
            st.subheader("Unique Results by Model")

            models_list = list(results.keys())
            model1, model2 = models_list[0], models_list[1]

            # Get unique identifiers from each model
            if not results[model1].empty and not results[model2].empty:
                model1_ids = set(row.get("title", "No Title") for _, row in results[model1].iterrows())
                model2_ids = set(row.get("title", "No Title") for _, row in results[model2].iterrows())

                # Unique results from each model
                model1_unique = model1_ids - model2_ids
                model2_unique = model2_ids - model1_ids

                # Create DataFrames with unique results sorted by ranking
                model1_unique_results = []
                for idx, row in enumerate(results[model1].itertuples()):
                    identifier = row.title
                    if identifier in model1_unique:
                        model1_unique_results.append(
                            {
                                "title": row.title,
                                "ranking": idx,
                                "distance": row.distance,
                            }
                        )

                model2_unique_results = []
                for idx, row in enumerate(results[model2].itertuples()):
                    identifier = row.title
                    if identifier in model2_unique:
                        model2_unique_results.append(
                            {
                                "title": row.title,
                                "ranking": idx,
                                "distance": row.distance,
                            }
                        )

                # Sort by ranking (position)
                model1_unique_results.sort(key=lambda x: x["ranking"])
                model2_unique_results.sort(key=lambda x: x["ranking"])

                # Show unique results from model 1
                if model1_unique_results:
                    st.markdown(
                        f"**🔍 {model1} - Unique Results (not in {model2}): {len(model1_unique_results)} total**"
                    )
                    for item in model1_unique_results:  # Show all unique results
                        with st.expander(f"📰 {item['title'][:80]}... (Ranking: #{item['ranking'] + 1})"):
                            st.markdown(f"**Ranking:** #{item['ranking'] + 1}")
                            st.markdown(f"**Distance:** {item['distance']:.4f}")

                            # Show description if available
                            if "description" in results[model1].columns:
                                desc_row = results[model1].iloc[item["ranking"]]
                                if "description" in desc_row and pd.notna(desc_row["description"]):
                                    st.markdown("**Description:**")
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                        <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                        {str(desc_row["description"])}
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                            # Show content if available
                            if "content" in results[model1].columns:
                                content_row = results[model1].iloc[item["ranking"]]
                                if "content" in content_row:
                                    st.markdown("**Content:**")
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                        <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                        {str(content_row["content"])}
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        # Show unique results from model 2
        if model2_unique_results:
            st.markdown(f"**🔍 {model2} - Unique Results (not in {model1}): {len(model2_unique_results)} total**")
            for item in model2_unique_results:  # Show all unique results
                with st.expander(f"📰 {item['title'][:80]}... (Ranking: #{item['ranking'] + 1})"):
                    st.markdown(f"**Ranking:** #{item['ranking'] + 1}")
                    st.markdown(f"**Distance:** {item['distance']:.4f}")

                    # Show description if available
                    if "description" in results[model2].columns:
                        desc_row = results[model2].iloc[item["ranking"]]
                        if "description" in desc_row and pd.notna(desc_row["description"]):
                            st.markdown("**Description:**")
                            st.markdown(
                                f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                        <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                        {str(desc_row["description"])}
                                        </div>
                                        </div>
                                        """,
                                unsafe_allow_html=True,
                            )

                    # Show content if available
                    if "content" in results[model2].columns:
                        content_row = results[model2].iloc[item["ranking"]]
                        if "content" in content_row:
                            st.markdown("**Content:**")
                            st.markdown(
                                f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                        <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                        {str(content_row["content"])}
                                        </div>
                                        </div>
                                        """,
                                unsafe_allow_html=True,
                            )

    def run_comparison(
        self,
        user_query: str,
        selected_models: List[str],
        sql_query: str,
        limit_results: int = 5,
        kwargs_json: str = "{}",
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes the same query with different models and returns the results.

        Args:
            user_query: User query
            selected_models: List of models to compare
            sql_query: SQL query to execute
            limit_results: Number of results per model
            kwargs_json: JSON string with additional parameters

        Returns:
            Dictionary with results by model
        """
        # Parse kwargs JSON
        kwargs = {}
        if kwargs_json and kwargs_json.strip() != "{}":
            try:
                import json

                kwargs = json.loads(kwargs_json)
                st.success(f"✅ kwargs parsed: {kwargs}")
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parsing JSON: {e}")
                st.info("Using empty kwargs")
                kwargs = {}

        results = {}

        for model in selected_models:
            try:
                df = self.semantic_search.search(
                    query=sql_query,
                    user_query=user_query,
                    embedding_type="_model_" + model,
                    model=model,
                    limit_results=limit_results,
                    **kwargs,
                )
                results[model] = df
            except Exception as e:
                st.error(f"Error with model {model}: {str(e)}")
                results[model] = pd.DataFrame()

        return results

    def run_dual(
        self,
        user_query: str,
        model_left: str,
        model_right: str,
        table_name_left: str,
        table_name_right: str,
        sql_query: str,
        limit_results: int = 5,
        min_date: str = "",
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes the same query for two models and returns side-by-side results.

        Defaults any empty model to "text_embedding".
        """
        # Default models
        model_left = model_left.strip() or "text_embedding"
        model_right = model_right.strip() or "text_embedding"

        results: Dict[str, pd.DataFrame] = {}

        pairs = [
            (model_left, table_name_left),
            (model_right, table_name_right),
        ]

        for model, table_name in pairs:
            try:
                # Per-side kwargs override for table_name
                side_kwargs = {}
                if table_name:
                    side_kwargs["table_name"] = table_name
                if min_date:
                    side_kwargs["min_date"] = min_date
                df = self.semantic_search.search(
                    query=sql_query,
                    user_query=user_query,
                    embedding_type="_model_" + model,
                    model=model,
                    limit_results=limit_results,
                    **side_kwargs,
                )
                results[table_name] = df
            except Exception as e:
                st.error(f"Error with model {model}: {str(e)}")
                results[table_name] = pd.DataFrame()

        return results

    def display_results(self, results: Dict[str, pd.DataFrame]):
        """
        Displays results in a comparative way in Streamlit.
        """
        if not results:
            st.warning("No results to display.")
            return

        # Show general metrics
        st.subheader("📊 Overall Metrics")
        col1, col2, col3 = st.columns(3)

        total_results = sum(len(df) for df in results.values() if not df.empty)
        with col1:
            st.metric("Total results across models", total_results)

        with col2:
            if any("distance" in df.columns for df in results.values() if not df.empty):
                all_distances = []
                for df in results.values():
                    if not df.empty and "distance" in df.columns:
                        all_distances.extend(df["distance"].tolist())
                if all_distances:
                    st.metric("Best distance", f"{min(all_distances):.4f}")

        with col3:
            st.metric("Models compared", len(results))

        # Show results by model in a single view
        for model_name, df in results.items():
            st.subheader(f"🔍 {model_name}")

            if df.empty:
                st.warning(f"No results obtained for {model_name}")
                continue

            # Model metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total results", len(df))
            with col2:
                if "distance" in df.columns:
                    st.metric("Min distance", f"{df['distance'].min():.4f}")
            with col3:
                if "distance" in df.columns:
                    st.metric("Max distance", f"{df['distance'].max():.4f}")

            # Show detailed results
            for idx, row in df.iterrows():
                # Create expander title with title and distance
                expander_title = f"Result {idx + 1}"
                if "title" in row and row["title"]:
                    expander_title += f" - {row['title']}"
                expander_title += f" (Distance: {row.get('distance', 'N/A'):.4f})"

                with st.expander(expander_title):
                    # Show title if exists
                    if "title" in row and row["title"]:
                        st.markdown(f"**Title:** {row['title']}")

                        # Show description if exists
                        if "description" in row and row["description"]:
                            st.markdown("**Description:**")
                            # Show complete description without scroll, extending downward
                            st.markdown(
                                f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                                <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                                {str(row["description"])}
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Show complete content in small font
                    if "content" in row:
                        st.markdown("**Content:**")
                        # Show complete content without scroll, extending downward
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                            <div style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;">
                            {str(row["content"])}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Show other metadata
                    metadata_cols = [
                        col for col in df.columns if col not in ["content", "distance", "title", "description"]
                    ]
                    if metadata_cols:
                        st.markdown("**Metadata:**")
                        for col in metadata_cols:
                            if pd.notna(row[col]):  # Only show if not NaN
                                st.write(f"**{col}:** {row[col]}")

    def launch_streamlit_app(self):
        """
        Lanza la aplicación Streamlit para comparar modelos.
        """
        st.set_page_config(page_title="Embedding Models Validator", page_icon="🔍", layout="wide")

        st.header("🔍 Embedding Models Validator")
        st.markdown("Compare different embedding models with the same query.")

        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")

            # User query
            user_query = st.text_area(
                "Search query:",
                value="trump tariffs economic impact",
                height=120,
                help="Enter the text you want to search for",
            )

            # Results limit
            limit_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

        # Top selectors in main area
        st.markdown("### Models per table")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            model_left = st.radio(
                "Left table model",
                options=self.available_models,
                index=self.available_models.index("text_embedding") if "text_embedding" in self.available_models else 0,
                horizontal=True,
                key="model_left_select",
            )
        with col_m2:
            model_right = st.radio(
                "Right table model",
                options=self.available_models,
                index=self.available_models.index("text_embedding") if "text_embedding" in self.available_models else 0,
                horizontal=True,
                key="model_right_select",
            )

        st.markdown("### Tables per model")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            table_name_left = st.text_input(
                "Left table_name", value="profiles_df_embeddings", placeholder="your_left_table"
            )
        with col_t2:
            table_name_right = st.text_input(
                "Right table_name", value="profiles_df_no_metadata_embeddings", placeholder="your_right_table"
            )

        # Optional min_date (example only)
        st.markdown("### Optional parameter")
        st.markdown("Enter JSON with only `min_date` (YYYY-MM-DD). Example:")

        kwargs_json = st.text_area(
            "kwargs (JSON) — example:",
            value='{"min_date": "2024-01-01"}',
            height=80,
            help='Only min_date is supported. Example: {"min_date": "2024-01-01"}',
        )

        # Main area
        st.subheader("📝 SQL Query")

        # Predefined query selector
        available_queries = self.get_available_queries()

        # Show information about available queries
        with st.expander("📁 Available queries"):
            st.markdown(
                f"""
            **Queries loaded from SQL files:**
            - Location: `src/queries/big_query/select_queries/`
            - Total loaded: {len(available_queries) - 1} (excluding Custom Query)

            **Files in queries_list:**
            {chr(10).join([f"- `{query}`" for query in queries_list])}
            """
            )

        names = list(available_queries.keys())
        default_idx = (
            names.index("Company Profiles Semantic Search") if "Company Profiles Semantic Search" in names else 0
        )
        query_option = st.selectbox("Select a predefined query:", names, index=default_idx)

        # Query editor
        sql_query = st.text_area(
            "SQL Query:",
            value=available_queries[query_option],
            height=200,
            help="You can modify the query. Use placeholders like {user_query}, {model}, {table_name}, {limit_results}.",
        )

        # Validate placeholders of the selected query
        if query_option != "Custom Query":
            validation = self.validate_query_placeholders(sql_query)
            missing_placeholders = [k for k, v in validation.items() if not v]

            if missing_placeholders:
                st.warning(f"⚠️ The selected query is missing these placeholders: {', '.join(missing_placeholders)}")
            else:
                st.success("✅ All required placeholders are present")

        # Extract optional min_date from JSON
        min_date_val = ""
        if kwargs_json and kwargs_json.strip():
            try:
                import json

                parsed = json.loads(kwargs_json)
                if isinstance(parsed, dict) and "min_date" in parsed:
                    min_date_val = str(parsed["min_date"]).strip()
            except Exception:
                min_date_val = ""

        # Execute button
        if st.button("🚀 Run Comparison", type="primary"):
            if not user_query:
                st.error("Please enter a search query.")
                return

            with st.spinner("Running comparison..."):
                results = self.run_dual(
                    user_query=user_query,
                    model_left=model_left,
                    model_right=model_right,
                    table_name_left=table_name_left,
                    table_name_right=table_name_right,
                    sql_query=sql_query,
                    limit_results=limit_results,
                    min_date=min_date_val,
                )

            st.subheader("📊 Results")

            # Side-by-side tables for immediate comparison
            left_df = results.get(table_name_left, pd.DataFrame())
            right_df = results.get(table_name_right, pd.DataFrame())

            # Flatten newline characters in preview tables so full text is visible
            def _flatten_newlines(df: pd.DataFrame) -> pd.DataFrame:
                if df is None or df.empty:
                    return df
                df = df.copy()
                print(f"df content {df.content.iloc[0]}")
                for col in ["content", "description"]:
                    if col in df.columns:
                        # Replace actual newlines and literal "\n" as text, and carriage returns
                        series = df[col].astype(str)
                        series = series.str.replace("\r", " ", regex=False)
                        series = series.str.replace("\n", " ", regex=False)
                        series = series.str.replace("\\n", " ", regex=True)
                        df[col] = series
                return df

            left_df_preview = _flatten_newlines(left_df.head(limit_results) if not left_df.empty else left_df)
            right_df_preview = _flatten_newlines(right_df.head(limit_results) if not right_df.empty else right_df)

            col_left, col_right = st.columns(2)
            with col_left:
                header_left = f"🔎 {model_left}"
                if table_name_left:
                    header_left += f" · {table_name_left}"
                st.subheader(header_left)
                if left_df.empty:
                    st.warning(f"No results for {model_left}")
                else:
                    st.dataframe(left_df_preview, use_container_width=True)
            with col_right:
                header_right = f"🔎 {model_right}"
                if table_name_right:
                    header_right += f" · {table_name_right}"
                st.subheader(header_right)
                if right_df.empty:
                    st.warning(f"No results for {model_right}")
                else:
                    st.dataframe(right_df_preview, use_container_width=True)

            # Detailed view below (expanders and metadata)
            self.display_results(results)

            # Show side-by-side comparison
            if len(results) > 1:
                st.subheader("⚖️ Distance Comparison")

                # Create comparison DataFrame
                comparison_data = []
                for model, df in results.items():
                    if not df.empty and "distance" in df.columns:
                        for idx, distance in enumerate(df["distance"].head(limit_results)):
                            comparison_data.append({"Model": model, "Ranking": idx + 1, "Distance": distance})

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)

                    # Line chart
                    import altair as alt

                    chart = (
                        alt.Chart(comparison_df)
                        .mark_line(point=True)
                        .encode(
                            x="Ranking:O", y="Distance:Q", color="Model:N", tooltip=["Model", "Ranking", "Distance"]
                        )
                        .properties(width=600, height=400, title="Distance Comparison by Ranking")
                    )

                    st.altair_chart(chart, use_container_width=True)

            # Typical embedding metrics
            if len(results) > 1:
                st.subheader("📈 Embedding Metrics")
                self.display_embedding_metrics(results)


def main():
    """
    Main function to run the validation app.
    """
    validator = EmbeddingValidator()
    validator.launch_streamlit_app()


if __name__ == "__main__":
    main()
