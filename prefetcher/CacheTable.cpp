#include "CacheTable.h"


/***
 * Constructs a cache which takes and returns data encapsulated as pointers to TupleRow or PyObject
 * Follows Least Recently Used replacement strategy
 * @param size Max elements the cache will hold, afterwards replacement takes place
 * @param table Name of the table being represented
 * @param keyspace Name of the keyspace whose table belongs to
 * @param query Query ready to be bind with the keys
 * @param session
 */
CacheTable::CacheTable(uint32_t size, const std::string &table,const std::string &keyspace,
                       const std::vector<std::string> &keyn,
                       const std::vector<std::string> &columns_n,
                       const std::string &token_range_pred,
                       const std::vector<std::pair<int64_t, int64_t>> &tkns,
                       CassSession *session) {
    columns_names = columns_n;
    key_names = keyn;
    tokens=tkns;
    token_predicate = "FROM " + keyspace + "." + table + " " + token_range_pred;
    get_predicate = "FROM " + keyspace + "." + table + " WHERE " + key_names[0] + "=?";
    select_keys = "SELECT " + key_names[0];
    for (uint16_t i = 1; i < key_names.size(); i++) {
        get_predicate += " AND " + key_names[i] + "=?";
        select_keys += "," + key_names[i];
    }

    select_values = columns_names[0];
    for (uint16_t i = 1; i < columns_names.size(); i++) {
        select_values += "," + columns_names[i];
    }
    select_values+=" ";
    select_keys+=" ";

    select_all = select_keys + "," + select_values;

    select_values = "SELECT " + select_values;

    this->myCache = new Poco::LRUCache<TupleRow, TupleRow>(size);
    cache_query = select_values + get_predicate;
    CassFuture *future = cass_session_prepare(session, cache_query.c_str());
    CassError rc = cass_future_error_code(future);
    assert(rc == CASS_OK && "can't connect");
    prepared_query = cass_future_get_prepared(future);

    this->session = session;
    cass_future_free(future);
    //lacks free future (future)
    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    assert(schema_meta != NULL && "error on schema");
    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace.c_str());

    assert(keyspace_meta != NULL && "error on keyspace");
    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table.c_str());


    all_names.reserve(key_names.size() + columns_names.size());
    all_names.insert(all_names.end(), key_names.begin(), key_names.end());
    all_names.insert(all_names.end(), columns_names.begin(), columns_names.end());

    keys_factory = new TupleRowFactory(table_meta, key_names);
    values_factory = new TupleRowFactory(table_meta, columns_names);
    items_factory = new TupleRowFactory(table_meta, all_names);
    cass_schema_meta_free(schema_meta);


};


CacheTable::~CacheTable() {
    cass_prepared_free(prepared_query);
    //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
    myCache->clear();
    delete(myCache);
    delete (keys_factory);
    delete (values_factory);
    delete (items_factory);
    session=NULL;
}




Prefetch* CacheTable::get_keys_iter(uint32_t prefetch_size) {
    return new Prefetch(&tokens, prefetch_size, keys_factory, session, select_keys + token_predicate);
}

Prefetch* CacheTable::get_values_iter(uint32_t prefetch_size) {
    return new Prefetch(&tokens, prefetch_size, values_factory, session, select_values + token_predicate);
}

Prefetch* CacheTable::get_items_iter(uint32_t prefetch_size) {
    return new Prefetch(&tokens, prefetch_size, items_factory, session, select_all + token_predicate);
}





void CacheTable::bind_keys(CassStatement *statement, TupleRow *keys) {

    for (uint16_t i = 0; i < keys->n_elem(); ++i) {
        const void *key = keys->get_element(i);
        uint16_t bind_pos = i;
        switch (keys_factory->get_type(i)) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                const char *temp = static_cast<const char *>(key);
                cass_statement_bind_string(statement, bind_pos, temp);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                //cass_statement_bind_bytes(statement,bind_pos,key,n_elem);
                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                cass_statement_bind_bool(statement, bind_pos, b);
                break;
            }
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                cass_statement_bind_double(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                cass_statement_bind_float(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                cass_statement_bind_int32(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TIMESTAMP: {

                break;
            }
            case CASS_VALUE_TYPE_UUID: {

                break;
            }
            case CASS_VALUE_TYPE_TIMEUUID: {

                break;
            }
            case CASS_VALUE_TYPE_INET: {

                break;
            }
            case CASS_VALUE_TYPE_DATE: {

                break;
            }
            case CASS_VALUE_TYPE_TIME: {

                break;
            }
            case CASS_VALUE_TYPE_SMALL_INT: {
                const int16_t *data = static_cast<const int16_t *>(key);
                cass_statement_bind_int16(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                cass_statement_bind_int8(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_LIST: {
                break;
            }
            case CASS_VALUE_TYPE_MAP: {

                break;
            }
            case CASS_VALUE_TYPE_SET: {

                break;
            }
            case CASS_VALUE_TYPE_TUPLE: {
                break;
            }
            default://CASS_VALUE_TYPE_UDT|CASS_VALUE_TYPE_CUSTOM|CASS_VALUE_TYPE_UNKNOWN:
                break;
        }
    }

}


/***
 * This method converts a python object (list) and converts it to a C++ TupleRow
 * and inserts it into the Cache.
 * The object is then put in a queue for being inserted into Cassandra.
 * @param row
 * @return
 */

void CacheTable::put_row(PyObject *key, PyObject *value) {
    TupleRow *k = keys_factory->make_tuple(key);
    TupleRow *v = values_factory->make_tuple(value);
    //Inserts if not present, otherwise replaces
    myCache->update(*k, v);
    delete (k);

}


/***
 * This method  does:
 * 1. converts the python object to C
 * 2. check if it is present in cache. If yes, returns it
 * 3. otherwise, queries Cassandra
 * 4. If a value is returned, it is put into the Cache, converted to Python and returned to the user.
 *
 * //TODO to be moved somewhere else :)
 * @param py_keys
 * @return
 */
PyObject *CacheTable::get_row(PyObject *py_keys) {

    TupleRow *keys = keys_factory->make_tuple(py_keys);

    Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
    if (!ptrElem.isNull()) {
        delete (keys);
        return values_factory->tuple_as_py(ptrElem.get());
    }
    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    bind_keys(statement, keys);

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        printf("%s\n", cass_error_desc(rc));
        cass_future_free(query_future);
        return NULL;
    }

    cass_future_free(query_future);
    cass_statement_free(statement);
    if (0 == cass_result_row_count(result)) {
        PyErr_SetString(PyExc_KeyError,"key not found");
        return NULL;
    }

    const CassRow *row = cass_result_first_row(result);

    //Store result to cache
    TupleRow *values = values_factory->make_tuple(row);

    myCache->add(*keys, values);
    delete (keys);
    cass_result_free(result);
    return values_factory->tuple_as_py(values);
}


//https://github.com/pocoproject/poco/blob/develop/Foundation/include/Poco/AbstractCache.h#L218
//Cache allocates space for each insert