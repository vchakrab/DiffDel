#fetch database stats
def get_database_statistics():
    """
    Connects to PostgreSQL and retrieves general database statistics.
    Users should customize the SQL query to fetch statistics relevant to their analysis.
    """
    stats = {}
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # SQL query to retrieve general database statistics
        # This is a placeholder. You need to customize this query
        # to get the specific statistics relevant to your analysis
        # as described in the uploaded paper or your use case.
        query = """
            SELECT
                (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') AS total_tables,
                (SELECT count(*) FROM pg_views WHERE schemaname = 'public') AS total_views,
                (SELECT count(*) FROM pg_sequences WHERE schemaname = 'public') AS total_sequences,
                (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public') AS total_indexes,
                pg_size_pretty(pg_database_size(current_database())) AS database_size;
        """
        cur.execute(query)
        result = cur.fetchone()

        if result:
            stats = {
                "total_tables": result[0],
                "total_views": result[1],
                "total_sequences": result[2],
                "total_indexes": result[3],
                "database_size": result[4]
            }

    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or fetching database statistics: {error}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")

    return stats

#Example usage:
if __name__ == "__main__":
     # Example for get_database_statistics
     db_statistics = get_database_statistics()

     if db_statistics:
         print(f"\nGeneral Database Statistics for '{db_params['database']}':")
         for stat_name, stat_value in db_statistics.items():
             print(f"  - {stat_name.replace('_', ' ').title()}: {stat_value}")
     else:
         print(f"\nCould not retrieve database statistics or an error occurred.")