from __future__ import annotations

from beanie import PydanticObjectId

from src.models.common import PipelineStatus
from src.models.feedback import Feedback
from src.models.material import Material
from src.models.pipeline_job import PipelineJob
from src.models.query_log import EmbeddedFeedback, QueryLog
from src.schemas.admin import AdminMetricsResponse, FeedbackRequest, FeedbackResponse, QueryStats, RetrievalStats


class AdminService:
    async def metrics(self) -> AdminMetricsResponse:
        total_docs = await Material.count()
        indexed_docs = await Material.find(Material.status == PipelineStatus.INDEXED.value).count()
        failed_jobs = await PipelineJob.find(PipelineJob.status == PipelineStatus.FAILED.value).count()
        feedback_count = await Feedback.count()

        # Use MongoDB aggregation to avoid loading all query logs into RAM
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "refused": {"$sum": {"$cond": ["$was_refused", 1, 0]}},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_latency": {"$avg": "$latency_ms"},
                    "avg_top_k": {"$avg": "$retrieval_trace.top_k"},
                    "avg_sources": {"$avg": "$retrieval_trace.sources_used_count"},
                    "avg_retrieval_ms": {"$avg": "$retrieval_trace.retrieval_time_ms"},
                }
            }
        ]
        agg_result = await QueryLog.aggregate(pipeline).to_list()
        row = agg_result[0] if agg_result else {}

        total_queries = row.get("total", 0)
        refused_queries = row.get("refused", 0)
        average_confidence = round(row.get("avg_confidence") or 0.0, 4)
        average_latency = round(row.get("avg_latency") or 0.0, 4)
        average_top_k = round(row.get("avg_top_k") or 0.0, 4)
        average_sources = round(row.get("avg_sources") or 0.0, 4)
        average_retrieval_ms = round(row.get("avg_retrieval_ms") or 0.0, 4)

        return AdminMetricsResponse(
            total_docs=total_docs,
            failed_jobs=failed_jobs,
            indexed_docs=indexed_docs,
            query_stats=QueryStats(
                total_queries=total_queries,
                refused_queries=refused_queries,
                average_confidence=average_confidence,
                average_latency_ms=average_latency,
            ),
            retrieval_stats=RetrievalStats(
                average_top_k=average_top_k,
                average_sources_used=average_sources,
                average_retrieval_time_ms=average_retrieval_ms,
            ),
            feedback_count=feedback_count,
        )

    async def log_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        query_log = await QueryLog.get(PydanticObjectId(request.query_log_id))
        if query_log is None or query_log.owner_id != request.owner_id:
            raise LookupError("Query log was not found for this owner")

        feedback = Feedback(
            owner_id=request.owner_id,
            query_log_id=query_log.id,
            rating=request.rating,
            comment=request.comment,
        )
        await feedback.insert()
        query_log.feedback.append(EmbeddedFeedback(rating=request.rating, comment=request.comment))
        await query_log.save()
        return FeedbackResponse(feedback_id=str(feedback.id), query_log_id=str(query_log.id), rating=feedback.rating)
