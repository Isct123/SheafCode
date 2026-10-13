                                                 
  
                                                                         
                                                                         
                                                                         
                                                                           
                                                                            
                                                                            
                                                  
  
                                            
  
                                                                             
                                                                                
                                                                            
                                                                                
                                    
  
                                                                               
                    
  
                                                                 
                                      
  
                                          
                                                                             
                                                                          
  
                                                                               
                                                                              
                                                      
                                          
  
                                                                              
                                        
                                                                             
                                                                         
                                                                              
                             
  
                                                                                
                                                                
  
                                                                           
                                                                          
                                                                      
                                                   
  
                                                                             
                                                                     
                                                                              
                                                                              
                                                                        
                                                                              
                                                                                
                                                                              
                                                                             
                                                                              
                                                                                
                                                                          
                                                                           
  
                                                                              
                                                                       
                                                                          
                                                                               
                                                            
  
                                                                           
                                                                       
                                                                                
                     
  
                                                                               
                                                                            
                                                                                
                                                                             
                                                                                
                                                                                
                                                                                
                                                                               
                                                                     
                                                                         
  
                                                                               
                                                                
  
                                                                           
                                                                          
                                                                         
                                                                               
                                                                             
                                      
                                                                        
                                                                            
            
                                                                                
                                                                             
                                                                              
                                                                           
                                                                                
                                                                                
                                                             
                                                           
                                                 
                                                   
                                                
                                                         
                                                                   
                                                                             
                                          
                                     
                                                                                             
                                                                                           
                                                        
                                                                    
                                                                         
                                                                           
                                                    
                                                                                       
                                                                                
                                                           
                                                                               
                                                                         
                                                                  
                                                                               
                                                                            
                                                                        
                                                                   
                                                                         
                                                                                
                                                                  
                                                                                
                                                  
                                                                              
                                                                           
                                                                              
                                                                              
                                                                               
                                                                
                                                                               
                                                                        
                                                                          
                                                                  
                                                                             
                                                                      
                                                                
                                                                        
                                                                        
                                                                                
                 
                                                                                
                        
  
          
                     
                                                         
                           
            
            
                                       
                                 
  
                                      
                                                           
  
                                          
                                     
                 
                                     
    
  
                                                           
              
            
                     
                 
                      
                    
    
  
                                                    
              
                          
                                           
                              
    
    
importUKAQ <-
  function(
    site = "my1",
    year = 2022,
    source = NULL,
    data_type = "hourly",
    pollutant = "all",
    hc = FALSE,
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
                               
    if ("local" %in% source) {
      cli::cli_warn(
        c(
          "i" = "'local' data is associated with locally managed air quality network sites in England.",
          "!" = "These sites are not part of the AURN national network, and therefore may not have the same level of quality control applied to them."
        ),
        .frequency = "regularly",
        .frequency_id = "lmam"
      )
    }

                   
    if (is.null(source)) {
      if (data_type %in% c("annual", "monthly", "daqi") && missing(site)) {
        cli::cli_abort(
          "Please provide a {.field source} when {.field data_type} is '{data_type}'."
        )
      }
      source <- guess_source(site)
    }

                                             
    if (meta) {
      meta_opts <- c(
        "site_type",
        "latitude",
        "longitude",
        "zone",
        "agglomeration",
        "local_authority"
      )

      if ("local" %in% source) {
        meta_opts <- append(meta_opts, "provider")
      }

      meta_columns <-
        rlang::arg_match(
          meta_columns,
          meta_opts,
          multiple = TRUE
        )
    }

                                            
    url_domain <- dplyr::recode_values(
      source,
      "aurn" ~ "https://uk-air.defra.gov.uk/openair/R_data/",
      "aqe" ~ "https://airqualityengland.co.uk/assets/openair/R_data/",
      "saqn" ~ "https://www.scottishairquality.scot/openair/R_data/",
      "waqn" ~
        "https://airquality.gov.wales/sites/default/files/openair/R_data/",
      "ni" ~ "https://www.airqualityni.co.uk/openair/R_data/",
      "local" ~ "https://uk-air.defra.gov.uk/openair/LMAM/R_data/"
    )

    url_abbr <- dplyr::recode_values(
      source,
      "aurn" ~ "_AURN_",
      "aqe" ~ "_AQE_",
      "saqn" ~ "_SCOT_",
      "waqn" ~ "_WAQ_",
      "ni" ~ "_NI_",
      "local" ~ "_LMAM_"
    )

                          
    allowed_types <- c(
      "hourly",
      "daily",
      "monthly",
      "annual",
      "daqi",
      "15min",
      "15_min",
      "24_hour",
      "8_hour",
      "daily_max_8"
    )

                                                  
    if (data_type == "15_min") {
      data_type <- "15min"
    }

    if (!tolower(data_type) %in% allowed_types) {
      cli::cli_warn(
        c(
          "!" = "'{data_type}' not recognised. Setting {.arg data_type} to 'hourly'",
          "i" = "{.arg data_type} should be one of: {allowed_types}"
        )
      )

      data_type <- "hourly"
    }

                                    
    if (data_type %in% c("annual", "monthly")) {
      if (missing(site)) {
        site <- "all"
      }

                         
      source <- unique(source)
      files <- paste0(url_domain, "summary_", data_type, url_abbr)
      files <- unique(files)

                         
      aq_data <- purrr::pmap(
        tidyr::crossing(tidyr::nesting(files, source), year),
        readSummaryData,
        data_type = data_type,
        to_narrow = to_narrow,
        hc = hc,
        .progress = ifelse(progress, "Importing Statistics", FALSE)
      ) |>
        purrr::list_rbind()

      if (nrow(aq_data) == 0) {
        cli::cli_abort(
          "No data returned. Check {.arg site}, {.arg year} and {.arg source}.",
          call = NULL
        )
      }
    }

                                 
    if (data_type == "daqi") {
      if ("local" %in% source) {
        cli::cli_abort(c(
          "!" = "{.arg data_type} 'DAQI' is not available for locally managed networks"
        ))
      }
      if (missing(site)) {
        site <- "all"
      }

                         
      source <- unique(source)
      files <- paste0(url_domain, "annual_DAQI", url_abbr)
      files <- unique(files)

                   
      aq_data <-
        purrr::pmap(
          tidyr::crossing(tidyr::nesting(files, source), year),
          readDAQI,
          .progress = ifelse(progress, "Importing DAQI", FALSE)
        ) |>
        purrr::list_rbind()

      if (nrow(aq_data) == 0) {
        cli::cli_abort(
          "No data returned. Check {.arg site}, {.arg year} and {.arg source}.",
          call = NULL
        )
      }

      aq_data <-
        aq_data |>
        dplyr::tibble() |>
        dplyr::mutate(
          band = dplyr::case_when(
            .data$poll_index %in% 1:3 ~ "Low",
            .data$poll_index %in% 4:6 ~ "Moderate",
            .data$poll_index %in% 7:9 ~ "High",
            .data$poll_index == 10 ~ "Very High"
          ),
          band = factor(.data$band, c("Low", "Moderate", "High", "Very High")),
          .after = "poll_index"
        )
    }

                           
    if (!data_type %in% c("annual", "monthly", "daqi")) {
      site <- toupper(site)
      if (length(source) == 1) {
        source <- rep(source, times = length(site))
      } else if (length(source) != length(site)) {
        cli::cli_abort(
          "Length of {.arg source} ({length(source)}) not equal to 1 or length of {.arg site} ({length(site)})."
        )
      }

                                                        
      if ("local" %in% source) {
                                   
        pcodes <-
          importMeta("local", all = TRUE) |>
          dplyr::distinct(.data$site, .keep_all = TRUE) |>
          dplyr::select("code", "pcode")

                              
        site_info <-
          data.frame(
            code = site,
            source = source,
            url_data = url_domain
          ) |>
          dplyr::left_join(pcodes, by = "code") |>
          tidyr::crossing(year = year)
      } else {
        site_info <-
          data.frame(
            code = site,
            source = source,
            url_data = url_domain
          ) |>
          dplyr::mutate(pcode = rep(NA, times = length(site))) |>
          tidyr::crossing(year = year)
      }

      aq_data <-
        purrr::pmap(
          .l = site_info,
          .f = purrr::possibly(
            ~ readUKAQData(
              site = ..1,
              lmam_subfolder = ..4,
              year = ..5,
              data_type,
              pollutant = pollutant,
              hc = hc,
              to_narrow = to_narrow,
              source = ..2,
              url_data = ..3,
              verbose = verbose
            ),
            quiet = !verbose
          ),
          .progress = ifelse(progress, "Importing AQ Data", FALSE)
        ) |>
        purrr::list_rbind()

      if (nrow(aq_data) == 0) {
        cli::cli_abort(
          "No data returned. Check {.arg site}, {.arg year} and {.arg source}.",
          call = NULL
        )
      }

      if (ratified && data_type == "hourly") {
        aq_data <-
          add_ratified(
            aq_data = aq_data,
            source = source,
            to_narrow = to_narrow
          )
      }
    }

                                     
    if (data_type %in% c("annual", "monthly", "daqi")) {
      aq_data <-
        filter_site_pollutant(
          aq_data,
          site = site,
          pollutant = pollutant,
          to_narrow = to_narrow,
          data_type = data_type
        )
    }

                                     
    if (!meteo) {
      aq_data <- aq_data |>
        dplyr::select(-dplyr::any_of(c("ws", "wd", "air_temp")))
    }

                    
    if (meta) {
      aq_data <-
        add_meta(source = source, columns = meta_columns, aq_data)
    }

                              
    aq_data <- dplyr::arrange(aq_data, "source", "site")

                 
    return(dplyr::as_tibble(aq_data))
  }

                                                        
  
                                                                             
                                                                           
                                                                              
                                                                            
                                                                               
                                                                                
                                                       
  
                                                             
  
                                                                           
                                                                          
                                                                         
                                                          
  
                            
                             
           
                           
          
importAURN <-
  function(
    site = "my1",
    year = 2009,
    data_type = "hourly",
    pollutant = "all",
    hc = FALSE,
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      hc = hc,
      meta = meta,
      meta_columns = meta_columns,
      meteo = meteo,
      ratified = ratified,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "aurn"
    )
  }

                             
           
          
importAQE <-
  function(
    site = "yk13",
    year = 2018,
    data_type = "hourly",
    pollutant = "all",
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      meta = meta,
      meta_columns = meta_columns,
      meteo = meteo,
      ratified = ratified,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "aqe"
    )
  }

                             
           
          
importSAQN <-
  function(
    site = "gla4",
    year = 2009,
    data_type = "hourly",
    pollutant = "all",
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      meta = meta,
      meta_columns = meta_columns,
      meteo = meteo,
      ratified = ratified,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "saqn"
    )
  }

                             
           
          
importWAQN <-
  function(
    site = "card",
    year = 2018,
    data_type = "hourly",
    pollutant = "all",
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      meta = meta,
      meta_columns = meta_columns,
      meteo = meteo,
      ratified = ratified,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "waqn"
    )
  }

                             
           
          
importNI <-
  function(
    site = "bel0",
    year = 2018,
    data_type = "hourly",
    pollutant = "all",
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    meteo = TRUE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      meta = meta,
      meta_columns = meta_columns,
      meteo = meteo,
      ratified = ratified,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "ni"
    )
  }

                             
           
          
importLocal <-
  function(
    site = "ad1",
    year = 2018,
    data_type = "hourly",
    pollutant = "all",
    meta = FALSE,
    meta_columns = c("site_type", "latitude", "longitude"),
    to_narrow = FALSE,
    verbose = FALSE,
    progress = TRUE
  ) {
    if (missing(site) && data_type %in% c("annual", "monthly", "daqi")) {
      site <- "all"
    }

    importUKAQ(
      site = site,
      year = year,
      data_type = data_type,
      pollutant = pollutant,
      meta = meta,
      meta_columns = meta_columns,
      meteo = TRUE,
      ratified = FALSE,
      to_narrow = to_narrow,
      progress = progress,
      verbose = verbose,
      source = "local"
    )
  }
