import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle


with open("dataset/stast.pkl", "rb") as f:
    stats = pickle.load(f)

with open("dataset/stratified_stats.pkl", "rb") as f:
    stratified_stats = pickle.load(f)


if __name__ == "__main__":
    # 1. Total Questions by Language (Before vs. After)
    languages = list(stats["questions_by_language"].keys())
    initial_counts = [stats["questions_by_language"][lang] for lang in languages]
    final_counts = [
        stratified_stats["questions_by_language"].get(lang, 0) for lang in languages
    ]

    # Plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(languages))

    plt.bar(index, initial_counts, bar_width, label="Initial", color="blue")
    plt.bar(
        [i + bar_width for i in index],
        final_counts,
        bar_width,
        label="Final",
        color="orange",
    )

    plt.xlabel("Language")
    plt.ylabel("Number of Questions")
    plt.title("Total Questions by Language (Before vs. After)")
    plt.xticks([i + bar_width / 2 for i in index], languages, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Multimodal Questions by Language (Before vs. After)
    initial_multimodal_counts = [
        stats["multimodal_by_language"].get(lang, 0) for lang in languages
    ]
    final_multimodal_counts = [
        stratified_stats["multimodal_by_language"].get(lang, 0) for lang in languages
    ]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(index, initial_multimodal_counts, bar_width, label="Initial", color="blue")
    plt.bar(
        [i + bar_width for i in index],
        final_multimodal_counts,
        bar_width,
        label="Final",
        color="orange",
    )

    plt.xlabel("Language")
    plt.ylabel("Number of Multimodal Questions")
    plt.title("Multimodal Questions by Language (Before vs. After)")
    plt.xticks([i + bar_width / 2 for i in index], languages, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # 3. Proportion of Multimodal Questions (Before vs. After)
    # initial_total_questions = sum(stats["questions_by_language"].values())
    # final_total_questions = sum(stratified_stats["questions_by_language"].values())
    # initial_multimodal_prop = stats["multimodal_questions"] / initial_total_questions
    # final_multimodal_prop = (
    #     stratified_stats["multimodal_questions"] / final_total_questions
    # )

    # # Plot
    # plt.figure(figsize=(6, 6))
    # labels = ["Initial", "Final"]
    # sizes = [initial_multimodal_prop, final_multimodal_prop]
    # colors = ["blue", "orange"]

    # plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    # plt.title("Proportion of Multimodal Questions (Before vs. After)")
    # plt.show()

    # 4. Questions by Category (Before vs. After)
    categories = list(stats["questions_by_category"].keys())
    general_categories = list(stats["questions_by_general_category"].keys())
    initial_category_counts = [
        stats["questions_by_category"][cat] for cat in categories
    ]
    final_category_counts = [
        stratified_stats["questions_by_category"].get(cat, 0) for cat in categories
    ]

    # Plot
    plt.figure(figsize=(12, 6))
    index = range(len(categories))

    plt.bar(index, initial_category_counts, bar_width, label="Initial", color="blue")
    plt.bar(
        [i + bar_width for i in index],
        final_category_counts,
        bar_width,
        label="Final",
        color="orange",
    )

    plt.xlabel("Category")
    plt.ylabel("Number of Questions")
    plt.title("Questions by Category (Before vs. After)")
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Multimodal Questions by Category (Before vs. After)
    initial_multimodal_category_counts = [
        stats["multimodal_by_category"].get(cat, 0) for cat in categories
    ]
    final_multimodal_category_counts = [
        stratified_stats["multimodal_by_category"].get(cat, 0) for cat in categories
    ]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(
        index,
        initial_multimodal_category_counts,
        bar_width,
        label="Initial",
        color="blue",
    )
    plt.bar(
        [i + bar_width for i in index],
        final_multimodal_category_counts,
        bar_width,
        label="Final",
        color="orange",
    )

    plt.xlabel("Category")
    plt.ylabel("Number of Multimodal Questions")
    plt.title("Multimodal Questions by Category (Before vs. After)")
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Capping Effect on Multimodal Questions

    initial_multimodal_counts = [
        stats["multimodal_by_language"].get(lang, 0) for lang in languages
    ]
    final_multimodal_counts = [
        stratified_stats["multimodal_by_language"].get(lang, 0) for lang in languages
    ]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(initial_multimodal_counts, final_multimodal_counts, color="blue")
    plt.plot(
        [0, max(initial_multimodal_counts)],
        [0, max(initial_multimodal_counts)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Initial Multimodal Questions")
    plt.ylabel("Final Multimodal Questions")
    plt.title("Capping Effect on Multimodal Questions")
    plt.show()

    # 7. Category Retention After Capping
    retained_categories = set(stratified_stats["questions_by_category"].keys())
    all_categories = set(stats["questions_by_category"].keys())
    removed_categories = all_categories - retained_categories

    # Print results
    print("Retained Categories:")
    print(retained_categories)

    print("\nRemoved Categories:")
    print(removed_categories)

    # 8. Language Retention After Capping
    retained_languages = set(stratified_stats["questions_by_language"].keys())
    all_languages = set(stats["questions_by_language"].keys())
    removed_languages = all_languages - retained_languages

    # Print results
    print("Retained Languages:")
    print(retained_languages)

    print("\nRemoved Languages:")
    print(removed_languages)

    # Extract data

    initial_multimodal_category_counts = [
        stats["multimodal_by_general_category"].get(cat, 0)
        for cat in general_categories
    ]
    final_multimodal_category_counts = [
        stratified_stats["multimodal_by_general_category"].get(cat, 0)
        for cat in general_categories
    ]

    # Define bar width and index
    bar_width = 0.35
    index = range(len(general_categories))  # Use range for x-axis positions

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(
        index,
        initial_multimodal_category_counts,
        bar_width,
        label="Initial",
        color="blue",
    )
    plt.bar(
        [i + bar_width for i in index],
        final_multimodal_category_counts,
        bar_width,
        label="Final",
        color="orange",
    )

    plt.xlabel("Category")
    plt.ylabel("Number of Multimodal Questions")
    plt.title("Multimodal Questions by General Category (Before vs. After)")
    plt.xticks([i + bar_width / 2 for i in index], general_categories, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stacked bar plots per language and in each language we have topic distribution
    data = []
    for (language, category), count in stratified_stats[
        "questions_by_language_and_general_category"
    ].items():
        data.append({"Language": language, "Category": category, "Count": count})

    df = pd.DataFrame(data)
    sns.set_palette("pastel")

    # Create the stacked bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df, x="Language", y="Count", hue="Category", width=0.9, linewidth=2
    )

    # Add labels and title
    plt.title("Topic Distribution by Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # multimodal vs text
    multimodal_counts = stratified_stats["multimodal_by_language"]
    total_counts = stratified_stats["questions_by_language"]
    text_counts = stratified_stats["text_by_language"]

    # Combine into a DataFrame
    data = []
    for lang in total_counts:
        data.append(
            {
                "Language": lang,
                "Question Type": "Multimodal",
                "Count": multimodal_counts.get(lang, 0),
            }
        )
        data.append(
            {
                "Language": lang,
                "Question Type": "Unimodal",
                "Count": text_counts.get(lang, 0),
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df, x="Language", y="Count", hue="Question Type", width=0.9, linewidth=1.5
    )

    # Add labels and title
    plt.title("Question Type Distribution by Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(title="Question Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

    import code

    code.interact(local=locals())
