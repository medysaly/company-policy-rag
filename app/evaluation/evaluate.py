import json

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.embeddings.embedder import get_embeddings
from app.generation.llm import get_llm
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.vectorstore.store import get_vector_store


def run_evaluation(handbook_path: str, eval_dataset_path: str):
    """Run RAGAS evaluation on the RAG pipeline."""

    # 1. Load and ingest the handbook
    print("Loading and ingesting handbook...")
    docs = load_document(handbook_path)
    chunks = chunk_documents(docs)
    store = get_vector_store()
    store.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks")

    # 2. Load eval questions
    with open(eval_dataset_path) as f:
        eval_data = json.load(f)

    # 3. Run each question through the RAG pipeline
    print(f"\nRunning {len(eval_data)} evaluation questions...")
    samples = []

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Retrieve
        results = store.similarity_search(question, k=3)
        contexts = [doc.page_content for doc in results]

        # Generate answer
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based ONLY on the following context.

Context:
{context}

Question: {question}

Answer:"""
        )

        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        context_str = "\n\n".join(contexts)
        answer = chain.invoke({"context": context_str, "question": question})

        print(f"  Q: {question}")
        print(f"  A: {answer[:100]}...")

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)

    # 4. Run RAGAS evaluation
    print("\nRunning RAGAS evaluation...")
    eval_dataset = EvaluationDataset(samples=samples)

    llm_wrapper = LangchainLLMWrapper(get_llm())
    embeddings = get_embeddings()

    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=embeddings,
    )

    # 5. Print results
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    scores = results.to_pandas()
    for col in scores.columns:
        if scores[col].dtype == "float64":
            print(f"  {col:25s}: {scores[col].mean():.4f}")
    print("=" * 50)

    return results
