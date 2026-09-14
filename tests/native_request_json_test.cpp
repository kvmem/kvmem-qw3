#include "native_request_json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static void exact(const json &a,const json &b) {
    if (a.type()!=b.type() || a!=b || a.dump()!=b.dump()) throw std::runtime_error("JSON value/type mismatch");
    if (a.is_object()) for (auto it=a.begin();it!=a.end();++it) exact(it.value(),b.at(it.key()));
    if (a.is_array()) for (size_t i=0;i<a.size();++i) exact(a[i],b[i]);
}
static void check(const std::string &wire) {
    json a,b;std::string old_error,new_error;
    try {a=json::parse(wire);} catch (const json::exception &e) {old_error=e.what();}
    try {b=qw3::detail::parse_native_request_json(wire);} catch (const json::exception &e) {new_error=e.what();}
    if (old_error!=new_error) throw std::runtime_error("JSON acceptance/error mismatch: "+wire);
    if (old_error.empty()) exact(a,b);
}
int main(int argc,char **argv) {
    try {
        std::vector<std::string> cases={"null","true","false","[]","{}","0","-0","-0.0","1.0","1e-9999","1e9999",
            "9223372036854775808","18446744073709551616","-9223372036854775809", "{\"x\":1,\"x\":-0.0}",
            "\"\\u0000\\ud83d\\ude00中文\"","\"\\ud800\"","\"\\udc00\"","\"\\x20\"",
            "{\"x\":NaN}","{\"x\":+1}","[1,]","[01]","true false","{unquoted:1}","/* comment */ 1",
            std::string("\xef\xbb\xbf")+"{}",std::string(300,'[')+"1"+std::string(300,']'),
            std::string("\"\xc0\xaf\""),std::string("\"\xff\""),std::string("[\"x\",\"a")+char(1)+"b\"]"};
        for (const auto &s:cases) check(s);
        size_t boundary_cases=0;
        // Exercise every byte at and around vector boundaries. The reference
        // parser decides both valid strings and exact malformed-input errors.
        for (size_t offset : {0u,1u,7u,15u,16u,17u,31u,32u,33u,47u,63u,64u,65u,127u,128u,129u}) {
            for (unsigned byte=0;byte<256;++byte) {
                check("[\""+std::string(offset,'A')+char(byte)+std::string(180,'B')+"\"]");
                ++boundary_cases;
            }
            for (const std::string suffix : {"\\u0000", "\\ud83d\\ude00", "\\\"", "\\\\", "中文😀"}) {
                check("[\""+std::string(offset,'A')+suffix+std::string(180,'B')+"\"]");
                ++boundary_cases;
            }
        }
        const std::string long_string="{\"image\":\""+std::string(512,'A')+"\"}";
        for (size_t length=0;length<long_string.size();++length) {
            check(long_string.substr(0,length));
            ++boundary_cases;
        }
        std::mt19937 random(0);
        for (int i=0;i<2000;++i) {
            json value={{"id",random()},{"number",double(int32_t(random()))/17.0},
                {"text",std::string(random()%1000,'A')+"中文 😀 \" \\ \n"},
                {"values",json::array({nullptr,true,-int64_t(random()),uint64_t(random())<<32})}};
            auto wire=value.dump();check(wire);
            if (i%2) wire.resize(random()%wire.size());else wire.insert(random()%wire.size(),1,char(random()%32));
            check(wire);
        }
        std::cerr<<"PASS: "<<cases.size()+4000+boundary_cases<<" valid/malformed documents, exact values/types/errors\n";
        json rows=json::array();
        for (int arg=1;arg<argc;++arg) {
            std::ifstream f(argv[arg],std::ios::binary);if(!f)throw std::runtime_error("fixture missing");
            std::string wire{std::istreambuf_iterator<char>(f),{}};
            json expected=json::parse(wire);
            for (int repetition=0;repetition<3;++repetition) for (int phase=0;phase<2;++phase) {
                bool fast=phase^(repetition&1);
                auto start=Clock::now();json actual=fast?qw3::detail::parse_native_request_json(wire):json::parse(wire);auto finish=Clock::now();
                // Full fixture comparisons stay outside the timing interval.
                if(actual!=expected)throw std::runtime_error("recorded native request changed");
                double ms=std::chrono::duration<double,std::milli>(finish-start).count();
                rows.push_back({{"fixture",argv[arg]},{"bytes",wire.size()},{"fast",fast},{"repetition",repetition},{"parse_ms",ms}});
                std::cerr<<arg<<" fast="<<fast<<" ms="<<ms<<'\n';
            }
        }
        std::cout<<json{{"passed",true},{"corpus_cases",cases.size()+4000+boundary_cases},{"boundary_cases",boundary_cases},{"rows",rows}}.dump(2)<<'\n';
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
