INSERT OVERWRITE DIRECTORY 'gs://bigda25/output/summary_statistics'
STORED AS TEXTFILE
SELECT
    COUNT(*) AS total_reviews,
    ROUND(AVG(rating), 2) AS avg_rating,
    MIN(rating) AS min_rating,
    MAX(rating) AS max_rating,
    ROUND(AVG(review_length), 2) AS avg_review_length,
    MIN(review_length) AS min_review_length,
    MAX(review_length) AS max_review_length,
    ROUND(AVG(helpful_vote), 2) AS avg_helpful_votes,
    MIN(helpful_vote) AS min_helpful_votes,
    MAX(helpful_vote) AS max_helpful_votes
FROM amazon_reviews;


-- here is the query to find the total number of reviews and it will be saved in the bucket accordingly--
INSERT OVERWRITE DIRECTORY 'gs://bigda25/output/total_reviews'
STORED AS TEXTFILE
SELECT COUNT(*) AS total_reviews FROM amazon_reviews;


-- here is the query to find the distribution of helpful reviews and it will be saved in the bucket accordingly--
INSERT OVERWRITE DIRECTORY 'gs://bigda25/output/helpful_reviews_distribution'
STORED AS TEXTFILE
SELECT is_helpful, COUNT(*) AS count
FROM amazon_reviews
GROUP BY is_helpful;


-- here is the query to find the top 5 reviewed products and it will be saved in the bucket accordingly--
INSERT OVERWRITE DIRECTORY 'gs://bigda25/output/top_reviewed_products'
STORED AS TEXTFILE
SELECT asin, COUNT(*) AS total_reviews
FROM amazon_reviews
GROUP BY asin
ORDER BY total_reviews DESC
LIMIT 5;

-- here is the query to find the average monthly rating and it will be saved in the bucket accordingly--
INSERT OVERWRITE DIRECTORY 'gs://bigda25/output/avg_monthly_rating'
STORED AS TEXTFILE
SELECT YEAR(review_date) AS year,
       MONTH(review_date) AS month,
       AVG(rating) AS avg_rating
FROM amazon_reviews
GROUP BY YEAR(review_date), MONTH(review_date)
ORDER BY year, month;
