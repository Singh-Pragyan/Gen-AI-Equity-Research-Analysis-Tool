FALLBACK_STOCKS = {
    "Indian Stock Market": {
        "Large Cap": {
            "Technology": {
                ("Low", "Long Term"): ["INFY", "TCS", "HCLTECH", "WIPRO", "TECHM"],
                ("Low", "Medium Term"): ["INFY", "TCS", "HCLTECH", "WIPRO", "LTIM"],
                ("Low", "Short Term"): ["INFY", "TCS", "HCLTECH", "TECHM", "LTIM"],
                ("Moderate", "Long Term"): ["INFY", "TCS", "WIPRO", "TECHM", "LTIM"],
                ("Moderate", "Medium Term"): ["INFY", "TCS", "HCLTECH", "LTIM", "COFORGE"],
                ("Moderate", "Short Term"): ["TCS", "HCLTECH", "LTIM", "COFORGE", "PERSISTENT"],
                ("High", "Long Term"): ["INFY", "TCS", "LTIM", "COFORGE", "PERSISTENT"],
                ("High", "Medium Term"): ["HCLTECH", "LTIM", "COFORGE", "PERSISTENT", "MPHASIS"],
                ("High", "Short Term"): ["LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "BIRLASOFT"]
            },
            "Finance": {
                ("Low", "Long Term"): ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
                ("Low", "Medium Term"): ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
                ("Low", "Short Term"): ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "BAJFINANCE"],
                ("Moderate", "Long Term"): ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "BAJFINANCE"],
                ("Moderate", "Medium Term"): ["ICICIBANK", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "SBIN"],
                ("Moderate", "Short Term"): ["KOTAKBANK", "AXISBANK", "BAJFINANCE", "SBIN", "HDFCBANK"],
                ("High", "Long Term"): ["HDFCBANK", "ICICIBANK", "BAJFINANCE", "SBIN", "KOTAKBANK"],
                ("High", "Medium Term"): ["BAJFINANCE", "AXISBANK", "SBIN", "KOTAKBANK", "ICICIBANK"],
                ("High", "Short Term"): ["BAJFINANCE", "SBIN", "AXISBANK", "KOTAKBANK", "HDFCBANK"]
            },
            "Energy": {
                ("Low", "Long Term"): ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL"],
                ("Low", "Medium Term"): ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "IOC"],
                ("Low", "Short Term"): ["RELIANCE", "NTPC", "POWERGRID", "BPCL", "IOC"],
                ("Moderate", "Long Term"): ["RELIANCE", "ONGC", "NTPC", "BPCL", "IOC"],
                ("Moderate", "Medium Term"): ["RELIANCE", "NTPC", "POWERGRID", "BPCL", "IOC"],
                ("Moderate", "Short Term"): ["ONGC", "BPCL", "IOC", "NTPC", "RELIANCE"],
                ("High", "Long Term"): ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC"],
                ("High", "Medium Term"): ["BPCL", "IOC", "ONGC", "RELIANCE", "NTPC"],
                ("High", "Short Term"): ["BPCL", "IOC", "ONGC", "RELIANCE", "ADANIGREEN"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
                ("Low", "Medium Term"): ["SUNPHARMA", "DRREDDY", "CIPLA", "APOLLOHOSP", "DIVISLAB"],
                ("Low", "Short Term"): ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"],
                ("Moderate", "Long Term"): ["SUNPHARMA", "DRREDDY", "DIVISLAB", "APOLLOHOSP", "LUPIN"],
                ("Moderate", "Medium Term"): ["DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "LUPIN"],
                ("Moderate", "Short Term"): ["CIPLA", "DIVISLAB", "LUPIN", "APOLLOHOSP", "SUNPHARMA"],
                ("High", "Long Term"): ["SUNPHARMA", "DIVISLAB", "LUPIN", "APOLLOHOSP", "DRREDDY"],
                ("High", "Medium Term"): ["LUPIN", "DIVISLAB", "APOLLOHOSP", "DRREDDY", "CIPLA"],
                ("High", "Short Term"): ["LUPIN", "APOLLOHOSP", "DIVISLAB", "DRREDDY", "AUROPHARMA"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TITAN"],
                ("Low", "Medium Term"): ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
                ("Low", "Short Term"): ["HINDUNILVR", "ITC", "NESTLEIND", "TITAN", "DABUR"],
                ("Moderate", "Long Term"): ["HINDUNILVR", "ITC", "BRITANNIA", "TITAN", "DABUR"],
                ("Moderate", "Medium Term"): ["ITC", "NESTLEIND", "BRITANNIA", "TITAN", "DABUR"],
                ("Moderate", "Short Term"): ["BRITANNIA", "TITAN", "DABUR", "HINDUNILVR", "ITC"],
                ("High", "Long Term"): ["HINDUNILVR", "ITC", "TITAN", "DABUR", "BRITANNIA"],
                ("High", "Medium Term"): ["TITAN", "DABUR", "BRITANNIA", "ITC", "NESTLEIND"],
                ("High", "Short Term"): ["TITAN", "DABUR", "BRITANNIA", "MARICO", "GODREJCP"]
            },
            "Automobile": {
                ("Low", "Long Term"): ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
                ("Low", "Medium Term"): ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR"],
                ("Low", "Short Term"): ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR"],
                ("Moderate", "Long Term"): ["MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR"],
                ("Moderate", "Medium Term"): ["TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR", "MARUTI"],
                ("Moderate", "Short Term"): ["BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR", "TATAMOTORS", "MARUTI"],
                ("High", "Long Term"): ["MARUTI", "EICHERMOT", "TVSMOTOR", "BAJAJ-AUTO", "TATAMOTORS"],
                ("High", "Medium Term"): ["TVSMOTOR", "EICHERMOT", "TATAMOTORS", "BAJAJ-AUTO", "MARUTI"],
                ("High", "Short Term"): ["TVSMOTOR", "TATAMOTORS", "EICHERMOT", "ASHOKLEY", "ESCORTS"]
            }
        },
        "Mid Cap": {
            "Technology": {
                ("Low", "Long Term"): ["LTIM", "MINDTREE", "COFORGE", "PERSISTENT", "MPHASIS"],
                ("Low", "Medium Term"): ["LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "ZENSARTECH"],
                ("Low", "Short Term"): ["LTIM", "COFORGE", "PERSISTENT", "ZENSARTECH", "SONATSOFTW"],
                ("Moderate", "Long Term"): ["MINDTREE", "COFORGE", "PERSISTENT", "MPHASIS", "ZENSARTECH"],
                ("Moderate", "Medium Term"): ["COFORGE", "PERSISTENT", "MPHASIS", "ZENSARTECH", "SONATSOFTW"],
                ("Moderate", "Short Term"): ["PERSISTENT", "ZENSARTECH", "SONATSOFTW", "COFORGE", "LTIM"],
                ("High", "Long Term"): ["COFORGE", "PERSISTENT", "ZENSARTECH", "SONATSOFTW", "MPHASIS"],
                ("High", "Medium Term"): ["ZENSARTECH", "SONATSOFTW", "PERSISTENT", "COFORGE", "TANLA"],
                ("High", "Short Term"): ["SONATSOFTW", "TANLA", "ZENSARTECH", "KPITTECH", "BIRLASOFT"]
            },
            "Finance": {
                ("Low", "Long Term"): ["BAJFINANCE", "CHOLAFIN", "SUNDARMFIN", "MUTHOOTFIN", "LICHSGFIN"],
                ("Low", "Medium Term"): ["BAJFINANCE", "CHOLAFIN", "SUNDARMFIN", "LICHSGFIN", "MUTHOOTFIN"],
                ("Low", "Short Term"): ["BAJFINANCE", "CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN", "SUNDARMFIN"],
                ("Moderate", "Long Term"): ["CHOLAFIN", "SUNDARMFIN", "MUTHOOTFIN", "LICHSGFIN", "BAJFINANCE"],
                ("Moderate", "Medium Term"): ["BAJFINANCE", "SUNDARMFIN", "MUTHOOTFIN", "LICHSGFIN", "CHOLAFIN"],
                ("Moderate", "Short Term"): ["SUNDARMFIN", "MUTHOOTFIN", "LICHSGFIN", "CHOLAFIN", "BAJFINANCE"],
                ("High", "Long Term"): ["BAJFINANCE", "CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN", "SUNDARMFIN"],
                ("High", "Medium Term"): ["CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN", "SUNDARMFIN", "360ONE"],
                ("High", "Short Term"): ["MUTHOOTFIN", "LICHSGFIN", "360ONE", "CHOLAFIN", "IDFCFIRSTB"]
            },
            "Energy": {
                ("Low", "Long Term"): ["ADANIGREEN", "TATAPOWER", "JSWENERGY", "ADANIPOWER", "TORNTPOWER"],
                ("Low", "Medium Term"): ["ADANIGREEN", "TATAPOWER", "JSWENERGY", "TORNTPOWER", "ADANIPOWER"],
                ("Low", "Short Term"): ["TATAPOWER", "JSWENERGY", "TORNTPOWER", "ADANIPOWER", "ADANIGREEN"],
                ("Moderate", "Long Term"): ["ADANIGREEN", "JSWENERGY", "ADANIPOWER", "TORNTPOWER", "TATAPOWER"],
                ("Moderate", "Medium Term"): ["TATAPOWER", "JSWENERGY", "TORNTPOWER", "ADANIPOWER", "ADANIGREEN"],
                ("Moderate", "Short Term"): ["JSWENERGY", "TORNTPOWER", "ADANIPOWER", "TATAPOWER", "ADANIGREEN"],
                ("High", "Long Term"): ["ADANIGREEN", "TATAPOWER", "ADANIPOWER", "TORNTPOWER", "JSWENERGY"],
                ("High", "Medium Term"): ["ADANIPOWER", "TORNTPOWER", "JSWENERGY", "TATAPOWER", "SUZLON"],
                ("High", "Short Term"): ["TORNTPOWER", "ADANIPOWER", "SUZLON", "JSWENERGY", "INOXWIND"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["LUPIN", "AUROPHARMA", "ALKEM", "IPCALAB", "TORNTPHARM"],
                ("Low", "Medium Term"): ["LUPIN", "AUROPHARMA", "ALKEM", "TORNTPHARM", "IPCALAB"],
                ("Low", "Short Term"): ["LUPIN", "ALKEM", "IPCALAB", "TORNTPHARM", "AUROPHARMA"],
                ("Moderate", "Long Term"): ["AUROPHARMA", "ALKEM", "IPCALAB", "TORNTPHARM", "LUPIN"],
                ("Moderate", "Medium Term"): ["LUPIN", "IPCALAB", "TORNTPHARM", "AUROPHARMA", "ALKEM"],
                ("Moderate", "Short Term"): ["IPCALAB", "TORNTPHARM", "AUROPHARMA", "LUPIN", "ALKEM"],
                ("High", "Long Term"): ["LUPIN", "AUROPHARMA", "TORNTPHARM", "IPCALAB", "ALKEM"],
                ("High", "Medium Term"): ["TORNTPHARM", "AUROPHARMA", "IPCALAB", "LUPIN", "GLENMARK"],
                ("High", "Short Term"): ["AUROPHARMA", "GLENMARK", "IPCALAB", "LUPIN", "NATCOPHARM"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["ASIANPAINT", "BERGEPAINT", "DABUR", "MARICO", "GODREJCP"],
                ("Low", "Medium Term"): ["ASIANPAINT", "BERGEPAINT", "DABUR", "GODREJCP", "MARICO"],
                ("Low", "Short Term"): ["ASIANPAINT", "DABUR", "MARICO", "GODREJCP", "BERGEPAINT"],
                ("Moderate", "Long Term"): ["BERGEPAINT", "DABUR", "MARICO", "GODREJCP", "ASIANPAINT"],
                ("Moderate", "Medium Term"): ["ASIANPAINT", "MARICO", "GODREJCP", "BERGEPAINT", "DABUR"],
                ("Moderate", "Short Term"): ["MARICO", "GODREJCP", "BERGEPAINT", "DABUR", "ASIANPAINT"],
                ("High", "Long Term"): ["ASIANPAINT", "DABUR", "GODREJCP", "BERGEPAINT", "MARICO"],
                ("High", "Medium Term"): ["GODREJCP", "BERGEPAINT", "MARICO", "DABUR", "JYOTHYLAB"],
                ("High", "Short Term"): ["BERGEPAINT", "JYOTHYLAB", "MARICO", "DABUR", "EMAMILTD"]
            },
            "Automobile": {
                ("Low", "Long Term"): ["ASHOKLEY", "TVSMOTOR", "ESCORTS", "MRF", "BALKRISIND"],
                ("Low", "Medium Term"): ["ASHOKLEY", "TVSMOTOR", "ESCORTS", "BALKRISIND", "MRF"],
                ("Low", "Short Term"): ["TVSMOTOR", "ESCORTS", "BALKRISIND", "MRF", "ASHOKLEY"],
                ("Moderate", "Long Term"): ["ASHOKLEY", "ESCORTS", "MRF", "BALKRISIND", "TVSMOTOR"],
                ("Moderate", "Medium Term"): ["TVSMOTOR", "ESCORTS", "BALKRISIND", "MRF", "ASHOKLEY"],
                ("Moderate", "Short Term"): ["ESCORTS", "BALKRISIND", "MRF", "ASHOKLEY", "TVSMOTOR"],
                ("High", "Long Term"): ["ASHOKLEY", "TVSMOTOR", "BALKRISIND", "MRF", "ESCORTS"],
                ("High", "Medium Term"): ["BALKRISIND", "MRF", "ASHOKLEY", "TVSMOTOR", "OLECTRA"],
                ("High", "Short Term"): ["MRF", "OLECTRA", "ASHOKLEY", "TVSMOTOR", "SMLISUZU"]
            }
        },
        "Small Cap": {
            "Technology": {
                ("Low", "Long Term"): ["TANLA", "BIRLASOFT", "KPITTECH", "ZENSARTECH", "SONATSOFTW"],
                ("Low", "Medium Term"): ["TANLA", "BIRLASOFT", "KPITTECH", "SONATSOFTW", "ZENSARTECH"],
                ("Low", "Short Term"): ["TANLA", "KPITTECH", "SONATSOFTW", "ZENSARTECH", "BIRLASOFT"],
                ("Moderate", "Long Term"): ["BIRLASOFT", "KPITTECH", "ZENSARTECH", "SONATSOFTW", "TANLA"],
                ("Moderate", "Medium Term"): ["TANLA", "ZENSARTECH", "SONATSOFTW", "BIRLASOFT", "KPITTECH"],
                ("Moderate", "Short Term"): ["KPITTECH", "SONATSOFTW", "BIRLASOFT", "TANLA", "ZENSARTECH"],
                ("High", "Long Term"): ["TANLA", "BIRLASOFT", "SONATSOFTW", "ZENSARTECH", "KPITTECH"],
                ("High", "Medium Term"): ["SONATSOFTW", "BIRLASOFT", "TANLA", "ZENSARTECH", "KPITTECH"],
                ("High", "Short Term"): ["BIRLASOFT", "KPITTECH", "TANLA", "SONATSOFTW", "INTELLECT"]
            },
            "Finance": {
                ("Low", "Long Term"): ["360ONE", "CREDITACC", "IDFCFIRSTB", "KARURVYSYA", "RBLBANK"],
                ("Low", "Medium Term"): ["360ONE", "CREDITACC", "IDFCFIRSTB", "RBLBANK", "KARURVYSYA"],
                ("Low", "Short Term"): ["360ONE", "IDFCFIRSTB", "KARURVYSYA", "RBLBANK", "CREDITACC"],
                ("Moderate", "Long Term"): ["CREDITACC", "IDFCFIRSTB", "KARURVYSYA", "RBLBANK", "360ONE"],
                ("Moderate", "Medium Term"): ["360ONE", "KARURVYSYA", "RBLBANK", "CREDITACC", "IDFCFIRSTB"],
                ("Moderate", "Short Term"): ["IDFCFIRSTB", "RBLBANK", "CREDITACC", "360ONE", "KARURVYSYA"],
                ("High", "Long Term"): ["360ONE", "CREDITACC", "RBLBANK", "KARURVYSYA", "IDFCFIRSTB"],
                ("High", "Medium Term"): ["RBLBANK", "CREDITACC", "IDFCFIRSTB", "360ONE", "KARURVYSYA"],
                ("High", "Short Term"): ["CREDITACC", "IDFCFIRSTB", "RBLBANK", "360ONE", "UJJIVANSFB"]
            },
            "Energy": {
                ("Low", "Long Term"): ["SUZLON", "INOXWIND", "KPEL", "IWEL", "OILCOUNTUB"],
                ("Low", "Medium Term"): ["SUZLON", "INOXWIND", "KPEL", "OILCOUNTUB", "IWEL"],
                ("Low", "Short Term"): ["SUZLON", "KPEL", "IWEL", "OILCOUNTUB", "INOXWIND"],
                ("Moderate", "Long Term"): ["INOXWIND", "KPEL", "IWEL", "OILCOUNTUB", "SUZLON"],
                ("Moderate", "Medium Term"): ["SUZLON", "IWEL", "OILCOUNTUB", "INOXWIND", "KPEL"],
                ("Moderate", "Short Term"): ["KPEL", "OILCOUNTUB", "INOXWIND", "SUZLON", "IWEL"],
                ("High", "Long Term"): ["SUZLON", "INOXWIND", "OILCOUNTUB", "IWEL", "KPEL"],
                ("High", "Medium Term"): ["OILCOUNTUB", "INOXWIND", "SUZLON", "KPEL", "IWEL"],
                ("High", "Short Term"): ["INOXWIND", "SUZLON", "KPEL", "IWEL", "KSK"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["GLENMARK", "NATCOPHARM", "AJANTPHARM", "ERIS", "JBCHEPHARM"],
                ("Low", "Medium Term"): ["GLENMARK", "NATCOPHARM", "AJANTPHARM", "JBCHEPHARM", "ERIS"],
                ("Low", "Short Term"): ["GLENMARK", "AJANTPHARM", "ERIS", "JBCHEPHARM", "NATCOPHARM"],
                ("Moderate", "Long Term"): ["NATCOPHARM", "AJANTPHARM", "ERIS", "JBCHEPHARM", "GLENMARK"],
                ("Moderate", "Medium Term"): ["GLENMARK", "ERIS", "JBCHEPHARM", "NATCOPHARM", "AJANTPHARM"],
                ("Moderate", "Short Term"): ["AJANTPHARM", "JBCHEPHARM", "NATCOPHARM", "GLENMARK", "ERIS"],
                ("High", "Long Term"): ["GLENMARK", "NATCOPHARM", "JBCHEPHARM", "ERIS", "AJANTPHARM"],
                ("High", "Medium Term"): ["JBCHEPHARM", "NATCOPHARM", "GLENMARK", "ERIS", "AJANTPHARM"],
                ("High", "Short Term"): ["NATCOPHARM", "GLENMARK", "AJANTPHARM", "ERIS", "MOREPENLAB"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["JYOTHYLAB", "EMAMILTD", "BAJAJCON", "VSTIND", "RADICO"],
                ("Low", "Medium Term"): ["JYOTHYLAB", "EMAMILTD", "BAJAJCON", "RADICO", "VSTIND"],
                ("Low", "Short Term"): ["JYOTHYLAB", "BAJAJCON", "VSTIND", "RADICO", "EMAMILTD"],
                ("Moderate", "Long Term"): ["EMAMILTD", "BAJAJCON", "VSTIND", "RADICO", "JYOTHYLAB"],
                ("Moderate", "Medium Term"): ["JYOTHYLAB", "VSTIND", "RADICO", "EMAMILTD", "BAJAJCON"],
                ("Moderate", "Short Term"): ["BAJAJCON", "RADICO", "EMAMILTD", "JYOTHYLAB", "VSTIND"],
                ("High", "Long Term"): ["JYOTHYLAB", "EMAMILTD", "RADICO", "VSTIND", "BAJAJCON"],
                ("High", "Medium Term"): ["RADICO", "EMAMILTD", "JYOTHYLAB", "VSTIND", "BAJAJCON"],
                ("High", "Short Term"): ["EMAMILTD", "BAJAJCON", "JYOTHYLAB", "RADICO", "GILLETTE"]
            },
            "Automobile": {
                ("Low", "Long Term"): ["SMLISUZU", "ATULAUTO", "OLECTRA", "GABRIEL", "MUNJALSHOW"],
                ("Low", "Medium Term"): ["SMLISUZU", "ATULAUTO", "OLECTRA", "MUNJALSHOW", "GABRIEL"],
                ("Low", "Short Term"): ["SMLISUZU", "OLECTRA", "GABRIEL", "MUNJALSHOW", "ATULAUTO"],
                ("Moderate", "Long Term"): ["ATULAUTO", "OLECTRA", "GABRIEL", "MUNJALSHOW", "SMLISUZU"],
                ("Moderate", "Medium Term"): ["SMLISUZU", "GABRIEL", "MUNJALSHOW", "ATULAUTO", "OLECTRA"],
                ("Moderate", "Short Term"): ["OLECTRA", "MUNJALSHOW", "ATULAUTO", "SMLISUZU", "GABRIEL"],
                ("High", "Long Term"): ["SMLISUZU", "ATULAUTO", "MUNJALSHOW", "GABRIEL", "OLECTRA"],
                ("High", "Medium Term"): ["MUNJALSHOW", "ATULAUTO", "SMLISUZU", "GABRIEL", "OLECTRA"],
                ("High", "Short Term"): ["ATULAUTO", "OLECTRA", "SMLISUZU", "GABRIEL", "HINDMOTORS"]
            }
        }
    },
    "US Stock Market": {
        "Large Cap": {
            "Technology": {
                ("Low", "Long Term"): ["AAPL", "MSFT", "GOOGL", "CSCO", "INTC"],
                ("Low", "Medium Term"): ["AAPL", "MSFT", "GOOGL", "INTC", "ORCL"],
                ("Low", "Short Term"): ["AAPL", "MSFT", "CSCO", "INTC", "ORCL"],
                ("Moderate", "Long Term"): ["MSFT", "GOOGL", "CSCO", "INTC", "ORCL"],
                ("Moderate", "Medium Term"): ["AAPL", "GOOGL", "CSCO", "ORCL", "IBM"],
                ("Moderate", "Short Term"): ["GOOGL", "CSCO", "INTC", "ORCL", "IBM"],
                ("High", "Long Term"): ["AAPL", "MSFT", "ORCL", "IBM", "CSCO"],
                ("High", "Medium Term"): ["CSCO", "INTC", "ORCL", "IBM", "NVDA"],
                ("High", "Short Term"): ["NVDA", "AMD", "QCOM", "TSM", "ASML"]
            },
            "Finance": {
                ("Low", "Long Term"): ["JPM", "V", "MA", "BAC", "WFC"],
                ("Low", "Medium Term"): ["JPM", "V", "MA", "WFC", "GS"],
                ("Low", "Short Term"): ["JPM", "V", "BAC", "WFC", "GS"],
                ("Moderate", "Long Term"): ["V", "MA", "BAC", "WFC", "GS"],
                ("Moderate", "Medium Term"): ["JPM", "MA", "BAC", "GS", "C"],
                ("Moderate", "Short Term"): ["MA", "BAC", "WFC", "GS", "C"],
                ("High", "Long Term"): ["JPM", "V", "WFC", "GS", "C"],
                ("High", "Medium Term"): ["BAC", "GS", "C", "JPM", "V"],
                ("High", "Short Term"): ["GS", "C", "BAC", "WFC", "MS"]
            },
            "Energy": {
                ("Low", "Long Term"): ["XOM", "CVX", "COP", "SLB", "EOG"],
                ("Low", "Medium Term"): ["XOM", "CVX", "COP", "EOG", "OXY"],
                ("Low", "Short Term"): ["XOM", "CVX", "SLB", "EOG", "OXY"],
                ("Moderate", "Long Term"): ["CVX", "COP", "SLB", "EOG", "OXY"],
                ("Moderate", "Medium Term"): ["XOM", "COP", "EOG", "OXY", "HAL"],
                ("Moderate", "Short Term"): ["COP", "SLB", "EOG", "OXY", "HAL"],
                ("High", "Long Term"): ["XOM", "CVX", "EOG", "OXY", "HAL"],
                ("High", "Medium Term"): ["SLB", "EOG", "OXY", "HAL", "COP"],
                ("High", "Short Term"): ["OXY", "HAL", "COP", "EOG", "APA"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
                ("Low", "Medium Term"): ["JNJ", "PFE", "MRK", "LLY", "BMY"],
                ("Low", "Short Term"): ["JNJ", "PFE", "ABBV", "LLY", "BMY"],
                ("Moderate", "Long Term"): ["PFE", "MRK", "ABBV", "LLY", "BMY"],
                ("Moderate", "Medium Term"): ["JNJ", "MRK", "LLY", "BMY", "GILD"],
                ("Moderate", "Short Term"): ["MRK", "ABBV", "LLY", "BMY", "GILD"],
                ("High", "Long Term"): ["JNJ", "PFE", "LLY", "BMY", "GILD"],
                ("High", "Medium Term"): ["ABBV", "LLY", "BMY", "GILD", "MRK"],
                ("High", "Short Term"): ["LLY", "BMY", "GILD", "MRK", "AMGN"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["PG", "KO", "PEP", "WMT", "COST"],
                ("Low", "Medium Term"): ["PG", "KO", "PEP", "COST", "CL"],
                ("Low", "Short Term"): ["PG", "KO", "WMT", "COST", "CL"],
                ("Moderate", "Long Term"): ["KO", "PEP", "WMT", "COST", "CL"],
                ("Moderate", "Medium Term"): ["PG", "PEP", "COST", "CL", "KMB"],
                ("Moderate", "Short Term"): ["PEP", "WMT", "COST", "CL", "KMB"],
                ("High", "Long Term"): ["PG", "KO", "COST", "CL", "KMB"],
                ("High", "Medium Term"): ["WMT", "COST", "CL", "KMB", "PEP"],
                ("High", "Short Term"): ["COST", "CL", "KMB", "PEP", "TGT"]
            },
            "Industrials": {
                ("Low", "Long Term"): ["CAT", "DE", "UNP", "BA", "HON"],
                ("Low", "Medium Term"): ["CAT", "DE", "UNP", "HON", "MMM"],
                ("Low", "Short Term"): ["CAT", "UNP", "BA", "HON", "MMM"],
                ("Moderate", "Long Term"): ["DE", "UNP", "BA", "HON", "MMM"],
                ("Moderate", "Medium Term"): ["CAT", "UNP", "HON", "MMM", "LMT"],
                ("Moderate", "Short Term"): ["UNP", "BA", "HON", "MMM", "LMT"],
                ("High", "Long Term"): ["CAT", "DE", "HON", "MMM", "LMT"],
                ("High", "Medium Term"): ["BA", "HON", "MMM", "LMT", "UNP"],
                ("High", "Short Term"): ["MMM", "LMT", "BA", "HON", "GE"]
            }
        },
        "Mid Cap": {
            "Technology": {
                ("Low", "Long Term"): ["CRWD", "OKTA", "ZS", "MDB", "DDOG"],
                ("Low", "Medium Term"): ["CRWD", "OKTA", "ZS", "DDOG", "FTNT"],
                ("Low", "Short Term"): ["CRWD", "ZS", "MDB", "DDOG", "FTNT"],
                ("Moderate", "Long Term"): ["OKTA", "ZS", "MDB", "DDOG", "FTNT"],
                ("Moderate", "Medium Term"): ["CRWD", "ZS", "DDOG", "FTNT", "PANW"],
                ("Moderate", "Short Term"): ["ZS", "MDB", "DDOG", "FTNT", "PANW"],
                ("High", "Long Term"): ["CRWD", "OKTA", "DDOG", "FTNT", "PANW"],
                ("High", "Medium Term"): ["MDB", "DDOG", "FTNT", "PANW", "ZS"],
                ("High", "Short Term"): ["DDOG", "FTNT", "PANW", "ZS", "SMCI"]
            },
            "Finance": {
                ("Low", "Long Term"): ["SCHW", "TROW", "FITB", "RF", "HBAN"],
                ("Low", "Medium Term"): ["SCHW", "TROW", "FITB", "HBAN", "KEY"],
                ("Low", "Short Term"): ["SCHW", "FITB", "RF", "HBAN", "KEY"],
                ("Moderate", "Long Term"): ["TROW", "FITB", "RF", "HBAN", "KEY"],
                ("Moderate", "Medium Term"): ["SCHW", "FITB", "HBAN", "KEY", "ZION"],
                ("Moderate", "Short Term"): ["FITB", "RF", "HBAN", "KEY", "ZION"],
                ("High", "Long Term"): ["SCHW", "TROW", "HBAN", "KEY", "ZION"],
                ("High", "Medium Term"): ["RF", "HBAN", "KEY", "ZION", "FITB"],
                ("High", "Short Term"): ["HBAN", "KEY", "ZION", "FITB", "CFR"]
            },
            "Energy": {
                ("Low", "Long Term"): ["APA", "DVN", "HES", "MRO", "FANG"],
                ("Low", "Medium Term"): ["APA", "DVN", "HES", "FANG", "PXD"],
                ("Low", "Short Term"): ["APA", "HES", "MRO", "FANG", "PXD"],
                ("Moderate", "Long Term"): ["DVN", "HES", "MRO", "FANG", "PXD"],
                ("Moderate", "Medium Term"): ["APA", "HES", "FANG", "PXD", "VLO"],
                ("Moderate", "Short Term"): ["HES", "MRO", "FANG", "PXD", "VLO"],
                ("High", "Long Term"): ["APA", "DVN", "FANG", "PXD", "VLO"],
                ("High", "Medium Term"): ["MRO", "FANG", "PXD", "VLO", "HES"],
                ("High", "Short Term"): ["FANG", "PXD", "VLO", "HES", "PSX"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["BAX", "HOLX", "VTRS", "BIO", "CRL"],
                ("Low", "Medium Term"): ["BAX", "HOLX", "VTRS", "CRL", "DVA"],
                ("Low", "Short Term"): ["BAX", "VTRS", "BIO", "CRL", "DVA"],
                ("Moderate", "Long Term"): ["HOLX", "VTRS", "BIO", "CRL", "DVA"],
                ("Moderate", "Medium Term"): ["BAX", "VTRS", "CRL", "DVA", "PODD"],
                ("Moderate", "Short Term"): ["VTRS", "BIO", "CRL", "DVA", "PODD"],
                ("High", "Long Term"): ["BAX", "HOLX", "CRL", "DVA", "PODD"],
                ("High", "Medium Term"): ["BIO", "CRL", "DVA", "PODD", "VTRS"],
                ("High", "Short Term"): ["CRL", "DVA", "PODD", "VTRS", "ALGN"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["CLX", "K", "HRL", "CAG", "SJM"],
                ("Low", "Medium Term"): ["CLX", "K", "HRL", "SJM", "CPB"],
                ("Low", "Short Term"): ["CLX", "HRL", "CAG", "SJM", "CPB"],
                ("Moderate", "Long Term"): ["K", "HRL", "CAG", "SJM", "CPB"],
                ("Moderate", "Medium Term"): ["CLX", "HRL", "SJM", "CPB", "MKC"],
                ("Moderate", "Short Term"): ["HRL", "CAG", "SJM", "CPB", "MKC"],
                ("High", "Long Term"): ["CLX", "K", "SJM", "CPB", "MKC"],
                ("High", "Medium Term"): ["CAG", "SJM", "CPB", "MKC", "HRL"],
                ("High", "Short Term"): ["SJM", "CPB", "MKC", "HRL", "LW"]
            },
            "Industrials": {
                ("Low", "Long Term"): ["MAS", "NDSN", "PNR", "AOS", "LSTR"],
                ("Low", "Medium Term"): ["MAS", "NDSN", "PNR", "LSTR", "ITT"],
                ("Low", "Short Term"): ["MAS", "PNR", "AOS", "LSTR", "ITT"],
                ("Moderate", "Long Term"): ["NDSN", "PNR", "AOS", "LSTR", "ITT"],
                ("Moderate", "Medium Term"): ["MAS", "PNR", "LSTR", "ITT", "WTS"],
                ("Moderate", "Short Term"): ["PNR", "AOS", "LSTR", "ITT", "WTS"],
                ("High", "Long Term"): ["MAS", "NDSN", "LSTR", "ITT", "WTS"],
                ("High", "Medium Term"): ["AOS", "LSTR", "ITT", "WTS", "PNR"],
                ("High", "Short Term"): ["LSTR", "ITT", "WTS", "PNR", "HAYW"]
            }
        },
        "Small Cap": {
            "Technology": {
                ("Low", "Long Term"): ["SMCI", "IONQ", "VRNS", "S", "AI"],
                ("Low", "Medium Term"): ["SMCI", "IONQ", "VRNS", "AI", "RAMP"],
                ("Low", "Short Term"): ["SMCI", "VRNS", "S", "AI", "RAMP"],
                ("Moderate", "Long Term"): ["IONQ", "VRNS", "S", "AI", "RAMP"],
                ("Moderate", "Medium Term"): ["SMCI", "VRNS", "AI", "RAMP", "BOX"],
                ("Moderate", "Short Term"): ["VRNS", "S", "AI", "RAMP", "BOX"],
                ("High", "Long Term"): ["SMCI", "IONQ", "AI", "RAMP", "BOX"],
                ("High", "Medium Term"): ["S", "AI", "RAMP", "BOX", "VRNS"],
                ("High", "Short Term"): ["AI", "RAMP", "BOX", "VRNS", "PATH"]
            },
            "Finance": {
                ("Low", "Long Term"): ["CUBI", "EZPW", "GDOT", "NAVI", "PFS"],
                ("Low", "Medium Term"): ["CUBI", "EZPW", "GDOT", "PFS", "EGBN"],
                ("Low", "Short Term"): ["CUBI", "GDOT", "NAVI", "PFS", "EGBN"],
                ("Moderate", "Long Term"): ["EZPW", "GDOT", "NAVI", "PFS", "EGBN"],
                ("Moderate", "Medium Term"): ["CUBI", "GDOT", "PFS", "EGBN", "LOB"],
                ("Moderate", "Short Term"): ["GDOT", "NAVI", "PFS", "EGBN", "LOB"],
                ("High", "Long Term"): ["CUBI", "EZPW", "PFS", "EGBN", "LOB"],
                ("High", "Medium Term"): ["NAVI", "PFS", "EGBN", "LOB", "GDOT"],
                ("High", "Short Term"): ["PFS", "EGBN", "LOB", "GDOT", "UPST"]
            },
            "Energy": {
                ("Low", "Long Term"): ["SM", "TALO", "CRK", "VTLE", "TELL"],
                ("Low", "Medium Term"): ["SM", "TALO", "CRK", "TELL", "SBOW"],
                ("Low", "Short Term"): ["SM", "CRK", "VTLE", "TELL", "SBOW"],
                ("Moderate", "Long Term"): ["TALO", "CRK", "VTLE", "TELL", "SBOW"],
                ("Moderate", "Medium Term"): ["SM", "CRK", "TELL", "SBOW", "CPE"],
                ("Moderate", "Short Term"): ["CRK", "VTLE", "TELL", "SBOW", "CPE"],
                ("High", "Long Term"): ["SM", "TALO", "TELL", "SBOW", "CPE"],
                ("High", "Medium Term"): ["VTLE", "TELL", "SBOW", "CPE", "CRK"],
                ("High", "Short Term"): ["TELL", "SBOW", "CPE", "CRK", "MTDR"]
            },
            "Healthcare": {
                ("Low", "Long Term"): ["ARWR", "BEAM", "NTLA", "KURA", "VIR"],
                ("Low", "Medium Term"): ["ARWR", "BEAM", "NTLA", "VIR", "SANA"],
                ("Low", "Short Term"): ["ARWR", "NTLA", "KURA", "VIR", "SANA"],
                ("Moderate", "Long Term"): ["BEAM", "NTLA", "KURA", "VIR", "SANA"],
                ("Moderate", "Medium Term"): ["ARWR", "NTLA", "VIR", "SANA", "CRSP"],
                ("Moderate", "Short Term"): ["NTLA", "KURA", "VIR", "SANA", "CRSP"],
                ("High", "Long Term"): ["ARWR", "BEAM", "VIR", "SANA", "CRSP"],
                ("High", "Medium Term"): ["KURA", "VIR", "SANA", "CRSP", "NTLA"],
                ("High", "Short Term"): ["VIR", "SANA", "CRSP", "NTLA", "EDIT"]
            },
            "Consumer Goods": {
                ("Low", "Long Term"): ["YETI", "SHOO", "OXM", "CAL", "GIII"],
                ("Low", "Medium Term"): ["YETI", "SHOO", "OXM", "GIII", "MOV"],
                ("Low", "Short Term"): ["YETI", "OXM", "CAL", "GIII", "MOV"],
                ("Moderate", "Long Term"): ["SHOO", "OXM", "CAL", "GIII", "MOV"],
                ("Moderate", "Medium Term"): ["YETI", "OXM", "GIII", "MOV", "WWW"],
                ("Moderate", "Short Term"): ["OXM", "CAL", "GIII", "MOV", "WWW"],
                ("High", "Long Term"): ["YETI", "SHOO", "GIII", "MOV", "WWW"],
                ("High", "Medium Term"): ["CAL", "GIII", "MOV", "WWW", "OXM"],
                ("High", "Short Term"): ["GIII", "MOV", "WWW", "OXM", "VSCO"]
            },
            "Industrials": {
                ("Low", "Long Term"): ["AMRC", "BLDP", "PLUG", "RUN", "SPCE"],
                ("Low", "Medium Term"): ["AMRC", "BLDP", "PLUG", "SPCE", "FCEL"],
                ("Low", "Short Term"): ["AMRC", "PLUG", "RUN", "SPCE", "FCEL"],
                ("Moderate", "Long Term"): ["BLDP", "PLUG", "RUN", "SPCE", "FCEL"],
                ("Moderate", "Medium Term"): ["AMRC", "PLUG", "SPCE", "FCEL", "NOVA"],
                ("Moderate", "Short Term"): ["PLUG", "RUN", "SPCE", "FCEL", "NOVA"],
                ("High", "Long Term"): ["AMRC", "BLDP", "SPCE", "FCEL", "NOVA"],
                ("High", "Medium Term"): ["RUN", "SPCE", "FCEL", "NOVA", "PLUG"],
                ("High", "Short Term"): ["SPCE", "FCEL", "NOVA", "PLUG", "NKLA"]
            }
        }
    }
}