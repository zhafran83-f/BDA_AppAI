"""Initial prediction history. Frozen DDL; never import current ORM here."""

from alembic import op

revision = "0001_prediction_history"
down_revision = None
branch_labels = None
depends_on = None

TABLE_DDL = (
    r"""
CREATE TABLE public.fashion_predictions (
	class_index SMALLINT NOT NULL, 
	label TEXT NOT NULL, 
	p_apparel DOUBLE PRECISION NOT NULL, 
	p_accessories DOUBLE PRECISION NOT NULL, 
	p_footwear DOUBLE PRECISION NOT NULL, 
	p_personal_care DOUBLE PRECISION NOT NULL, 
	image_size_bytes INTEGER NOT NULL, 
	image_format TEXT NOT NULL, 
	id UUID NOT NULL, 
	predicted_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	stored_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	inference_duration_ms BIGINT NOT NULL, 
	model_version TEXT NOT NULL, 
	preprocessing_version TEXT NOT NULL, 
	artifact_sha256 JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_fashion_predictions_duration CHECK (inference_duration_ms >= 0), 
	CONSTRAINT ck_fashion_predictions_model_version CHECK (btrim(model_version) <> ''), 
	CONSTRAINT ck_fashion_predictions_preprocessing_version CHECK (btrim(preprocessing_version) <> ''), 
	CONSTRAINT ck_fashion_predictions_artifacts CHECK ((jsonb_typeof(artifact_sha256) = 'object' AND artifact_sha256 ?& ARRAY['vgg16'] AND artifact_sha256 - ARRAY['vgg16'] = '{}'::jsonb AND jsonb_typeof(artifact_sha256 -> 'vgg16') = 'string' AND (artifact_sha256 ->> 'vgg16') ~ '^[0-9a-f]{64}$') IS TRUE), 
	CONSTRAINT ck_fashion_predictions_class_label CHECK ((class_index = 0 AND label = 'Apparel') OR (class_index = 1 AND label = 'Accessories') OR (class_index = 2 AND label = 'Footwear') OR (class_index = 3 AND label = 'Personal Care')), 
	CONSTRAINT ck_fashion_predictions_image_size CHECK (image_size_bytes BETWEEN 1 AND 5242880), 
	CONSTRAINT ck_fashion_predictions_image_format CHECK (image_format IN ('JPEG', 'PNG')), 
	CONSTRAINT ck_fashion_predictions_probabilities_range CHECK (p_apparel BETWEEN 0 AND 1 AND p_accessories BETWEEN 0 AND 1 AND p_footwear BETWEEN 0 AND 1 AND p_personal_care BETWEEN 0 AND 1), 
	CONSTRAINT ck_fashion_predictions_probabilities_sum CHECK (abs((p_apparel + p_accessories + p_footwear + p_personal_care) - 1.0) <= 0.00001 * greatest(abs((p_apparel + p_accessories + p_footwear + p_personal_care)), 1.0))
)
""",
    r"""
CREATE TABLE public.review_clusters (
	cluster SMALLINT NOT NULL, 
	input_character_count INTEGER NOT NULL, 
	cleaned_character_count INTEGER NOT NULL, 
	id UUID NOT NULL, 
	predicted_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	stored_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	inference_duration_ms BIGINT NOT NULL, 
	model_version TEXT NOT NULL, 
	preprocessing_version TEXT NOT NULL, 
	artifact_sha256 JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_review_clusters_duration CHECK (inference_duration_ms >= 0), 
	CONSTRAINT ck_review_clusters_model_version CHECK (btrim(model_version) <> ''), 
	CONSTRAINT ck_review_clusters_preprocessing_version CHECK (btrim(preprocessing_version) <> ''), 
	CONSTRAINT ck_review_clusters_artifacts CHECK ((jsonb_typeof(artifact_sha256) = 'object' AND artifact_sha256 ?& ARRAY['tfidf', 'kmeans'] AND artifact_sha256 - ARRAY['tfidf', 'kmeans'] = '{}'::jsonb AND jsonb_typeof(artifact_sha256 -> 'tfidf') = 'string' AND (artifact_sha256 ->> 'tfidf') ~ '^[0-9a-f]{64}$' AND jsonb_typeof(artifact_sha256 -> 'kmeans') = 'string' AND (artifact_sha256 ->> 'kmeans') ~ '^[0-9a-f]{64}$') IS TRUE), 
	CONSTRAINT ck_review_clusters_cluster CHECK (cluster IN (0, 1)), 
	CONSTRAINT ck_review_clusters_input_length CHECK (input_character_count BETWEEN 1 AND 10000), 
	CONSTRAINT ck_review_clusters_cleaned_length CHECK (cleaned_character_count > 0)
)
""",
    r"""
CREATE TABLE public.purchase_predictions (
	rf_revenue_class SMALLINT NOT NULL, 
	svm_revenue_class SMALLINT NOT NULL, 
	mlp_revenue_class SMALLINT NOT NULL, 
	rf_probability_0 DOUBLE PRECISION NOT NULL, 
	rf_probability_1 DOUBLE PRECISION NOT NULL, 
	mlp_probability_0 DOUBLE PRECISION NOT NULL, 
	mlp_probability_1 DOUBLE PRECISION NOT NULL, 
	svm_probabilities JSONB, 
	id UUID NOT NULL, 
	predicted_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	stored_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	inference_duration_ms BIGINT NOT NULL, 
	model_version TEXT NOT NULL, 
	preprocessing_version TEXT NOT NULL, 
	artifact_sha256 JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_purchase_predictions_duration CHECK (inference_duration_ms >= 0), 
	CONSTRAINT ck_purchase_predictions_model_version CHECK (btrim(model_version) <> ''), 
	CONSTRAINT ck_purchase_predictions_preprocessing_version CHECK (btrim(preprocessing_version) <> ''), 
	CONSTRAINT ck_purchase_predictions_artifacts CHECK ((jsonb_typeof(artifact_sha256) = 'object' AND artifact_sha256 ?& ARRAY['random_forest', 'svm', 'mlp', 'feature_columns'] AND artifact_sha256 - ARRAY['random_forest', 'svm', 'mlp', 'feature_columns'] = '{}'::jsonb AND jsonb_typeof(artifact_sha256 -> 'random_forest') = 'string' AND (artifact_sha256 ->> 'random_forest') ~ '^[0-9a-f]{64}$' AND jsonb_typeof(artifact_sha256 -> 'svm') = 'string' AND (artifact_sha256 ->> 'svm') ~ '^[0-9a-f]{64}$' AND jsonb_typeof(artifact_sha256 -> 'mlp') = 'string' AND (artifact_sha256 ->> 'mlp') ~ '^[0-9a-f]{64}$' AND jsonb_typeof(artifact_sha256 -> 'feature_columns') = 'string' AND (artifact_sha256 ->> 'feature_columns') ~ '^[0-9a-f]{64}$') IS TRUE), 
	CONSTRAINT ck_purchase_predictions_classes CHECK (rf_revenue_class IN (0, 1) AND svm_revenue_class IN (0, 1) AND mlp_revenue_class IN (0, 1)), 
	CONSTRAINT ck_purchase_predictions_svm_probabilities CHECK (svm_probabilities IS NULL), 
	CONSTRAINT ck_purchase_predictions_rf_probabilities_range CHECK (rf_probability_0 BETWEEN 0 AND 1 AND rf_probability_1 BETWEEN 0 AND 1), 
	CONSTRAINT ck_purchase_predictions_rf_probabilities_sum CHECK (abs((rf_probability_0 + rf_probability_1) - 1.0) <= 0.00001 * greatest(abs((rf_probability_0 + rf_probability_1)), 1.0)), 
	CONSTRAINT ck_purchase_predictions_mlp_probabilities_range CHECK (mlp_probability_0 BETWEEN 0 AND 1 AND mlp_probability_1 BETWEEN 0 AND 1), 
	CONSTRAINT ck_purchase_predictions_mlp_probabilities_sum CHECK (abs((mlp_probability_0 + mlp_probability_1) - 1.0) <= 0.00001 * greatest(abs((mlp_probability_0 + mlp_probability_1)), 1.0))
)
""",
)

TABLES = (
    "fashion_predictions",
    "review_clusters",
    "purchase_predictions",
)


def upgrade():
    for statement in TABLE_DDL:
        op.execute(statement)
    for table in TABLES:
        op.execute(
            f"CREATE INDEX ix_{table}_history "
            f"ON public.{table} (predicted_at DESC, id DESC)"
        )
        # Remove inherited PUBLIC/default-ACL access on these NEW tables only.
        op.execute(
            f"REVOKE ALL PRIVILEGES ON TABLE public.{table} "
            "FROM PUBLIC, bda_app_runtime"
        )
        op.execute(
            f"GRANT SELECT, INSERT ON TABLE public.{table} "
            "TO bda_app_runtime"
        )
    op.execute(
        "REVOKE ALL PRIVILEGES ON TABLE public.alembic_version "
        "FROM PUBLIC, bda_app_runtime"
    )


def downgrade():
    # Destructive, explicit-only operation. Never run from application startup.
    for table in reversed(TABLES):
        op.drop_table(table, schema="public")
