#ifndef PREFETCHER_MY_TUPLE_H
#define PREFETCHER_MY_TUPLE_H

#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <iostream>
#include "stdlib.h"
#include <vector>
#include <cassandra.h>
#include <python2.7/Python.h>
#include "metadata.h"


class TupleRow {
private:
    std::shared_ptr<const void> payload;
    std::shared_ptr<std::vector<ColumnMeta>> metadata;
    uint16_t payload_size;
public:

    TupleRow(const std::shared_ptr<std::vector<ColumnMeta>> metas, uint16_t payload_size,void *buffer);

    TupleRow(const TupleRow &t) ;

    TupleRow(const TupleRow *t);

    TupleRow(TupleRow *t);

    TupleRow(TupleRow &t);

    TupleRow& operator=( const TupleRow& other );

    TupleRow& operator=(TupleRow& other );


    inline const uint16_t n_elem() const {
        return (uint16_t) metadata.get()->size();
    }

    const void* get_element(int32_t position) const {
        if (position < 0 || payload.get() == 0) return 0;
        return (const char *) payload.get() + metadata.get()->at(position).position;
    }

    friend bool operator<(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator<=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator==(const TupleRow &lhs, const TupleRow &rhs);


};

#endif //PREFETCHER_MY_TUPLE_H