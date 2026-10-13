                                                                
  
                                                                             
                                                                               
                                                                            
                                                                       
  
                               
  
                                                                             
            
  
                                                                                         
  
                                                                                       
  
                                                                                         
  
                                                                               
  
                                                                                          
  
                                                                                                        
                                     
  
                                                                                      
  
                                                                           
                                                                               
                                                                              
  
                                
  
                                                                                
                                                                       
                     
  
               
  
             
  
                                   
  
             
  
             
  
                  
  
                 
  
              
  
                                                                         
                                                                            
                                                                               
                                                                            
                                       
  
                                                                           
                                                                        
  
                            
  
                                                                               
                                                                                
                                                                           
             
  
                                                                         
                                                                                
                                                                
  
                                                                        
              
  
                                                                              
  
                                                          
  
                                                                      
                                                                       
  
                                                                          
                                                                              
                                                                               
                                                       
  
                                                              
                                                                          
  
                                                                         
                                                                          
                                                  
  
                                                                             
              
  
                                                                    
                                           
  
                                                                             
                                                 
  
                                                           
                                                 
                                                   
                                                
                                                         
                                                                   
                                                                                       
                                                     
  
                                                               
  
                                                                       
                                                                               
               
                                                                      
                                  
  
                                                                              
                                                                                
                                                                                
                                                                              
                                                             
  
                                                                               
                                                                       
                                                                                
                                                                               
                                                                               
                                                                              
                 
  
                                                                         
                                                                       
                                                                              
                                                                                
                                                                            
                                                                                
                                                  
  
                                                                              
                                                                          
                                                                         
                                                                            
                           
  
                                                                          
                                                                              
                                                                              
                                                                            
                                                                            
                                                                           
              
  
                                                                                
                                                                                
                                                                              
                                                      
  
                                                                            
                                                                               
                                                                               
                                      
  
                                                                               
                                                                              
                                                                               
  
                       
  
                        
                       
  
                           
  
                                                                                
                                                   
  
                                                                               
                                                                    
  
          
            
            
                                 
                                      
  
                              
                                                  
  
                           
                                                 
  
                                        
                                                                  
  
                                                  
                      
                                        
                    
                     
                 
    
    
importMeta <-
  function(
    source = "aurn",
    year = NULL,
    pollutant = NULL,
    code = NULL,
    site = NULL,
    site_type = NULL,
    lat = NULL,
    lng = NULL,
    crs = 4326,
    max_dist = NULL,
    max_n = NULL,
    all = FALSE,
    duplicate = FALSE
  ) {
                       
    source <- tolower(source)

                            
    source[source == "kcl"] <- "imperial"
    source[source == "saqd"] <- "saqn"
    source[source == "niaqn"] <- "ni"
    source[source == "lmam"] <- "local"

                         
    if (any(source == "ukaq")) {
      source <- c("aurn", "aqe", "saqn", "waqn", "ni", "local")
    }
    if (any(source == "all")) {
      source <-
        c("aurn", "aqe", "saqn", "waqn", "ni", "local", "imperial", "europe")
    }

                             
    source <- rlang::arg_match(
      source,
      c(
        "aurn",
        "imperial",
        "saqn",
        "waqn",
        "aqe",
        "local",
        "ni",
        "europe"
      ),
      multiple = TRUE
    )

                                                   
    get_meta <- function(source, all) {
      if (!source %in% c("imperial", "europe")) {
        url <- switch(
          source,
          aurn = "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData",
          saqn = "https://www.scottishairquality.scot/openair/R_data/SCOT_metadata.RData",
          saqd = "https://www.scottishairquality.scot/openair/R_data/SCOT_metadata.RData",
          ni = "https://www.airqualityni.co.uk/openair/R_data/NI_metadata.RData",
          waqn = "https://airquality.gov.wales/sites/default/files/openair/R_data/WAQ_metadata.RData",
          aqe = "https://airqualityengland.co.uk/assets/openair/R_data/AQE_metadata.RData",
          local = "https://uk-air.defra.gov.uk/openair/LMAM/R_data/LMAM_metadata.RData",
          lmam = "https://uk-air.defra.gov.uk/openair/LMAM/R_data/LMAM_metadata.RData"
        )

        meta <- read_wsp_meta(
          url,
          all = all,
          year = year,
          pollutant = pollutant
        )
      }

           
      if (source == "imperial") {
        meta <- read_imperial_meta(
          "https://londonair.org.uk/r_data/sites.RData",
          all = all,
          year = year
        )
      }

              
      if (source == "europe") {
        meta <- read_saqgetr_meta(
          "http://aq-data.ricardo-aea.com/R_data/saqgetr/helper_tables/sites.gz",
          year = year
        )
      }

                          
      meta <- dplyr::mutate(
        meta,
        source = source,
        .before = dplyr::everything()
      )

      return(meta)
    }

                      
    meta <-
      purrr::map(.x = source, .f = ~ get_meta(source = .x, all = all)) |>
      purrr::list_rbind() |>
      dplyr::tibble()

    if (nrow(meta) == 0) {
      warn_empty_meta()
      return(meta)
    }

                 
    if (!is.null(code) && "code" %in% names(meta)) {
      meta <- dplyr::filter(
        meta,
        tolower(.data$code) %in% tolower({{ code }})
      )

      if (nrow(meta) == 0) {
        warn_empty_meta()
        return(meta)
      }
    }

                      
    if (!is.null(site) && "site" %in% names(meta)) {
      all_sites <- paste(site, collapse = "|")
      meta <- dplyr::filter(
        meta,
        grepl(!!all_sites, .data$site, ignore.case = TRUE)
      )

      if (nrow(meta) == 0) {
        warn_empty_meta()
        return(meta)
      }
    }

                      
    if (!is.null(site_type) && "site_type" %in% names(meta)) {
      meta <- dplyr::filter(
        meta,
        tolower(.data$site_type) %in% tolower({{ site_type }})
      )

      if (nrow(meta) == 0) {
        warn_empty_meta()
        return(meta)
      }
    }

                                     
    if (!all) {
      meta <-
        dplyr::select(
          meta,
          dplyr::all_of(
            c(
              "source",
              "site",
              "code",
              "latitude",
              "longitude",
              "site_type"
            )
          )
        )
    }

                          
    if (!duplicate) {
      if (all && "variable" %in% names(meta)) {
        meta <- dplyr::distinct(
          meta,
          .data$code,
          .data$latitude,
          .data$longitude,
          .data$variable,
          .keep_all = TRUE
        )
      } else {
        meta <- dplyr::distinct(
          meta,
          .data$code,
          .data$latitude,
          .data$longitude,
          .keep_all = TRUE
        )
      }
    }

                       
    if (!is.null(lat) && !is.null(lng)) {
      target <- sf::st_as_sf(
        dplyr::tibble(lat = lat, lng = lng),
        coords = c("lng", "lat"),
        crs = crs
      ) |>
        sf::st_transform(crs = 4326)

      meta_old <- meta
      meta <- tidyr::drop_na(meta, dplyr::all_of(c("latitude", "longitude")))

      if (nrow(meta) != nrow(meta_old)) {
        cli::cli_warn(
          "Dropping {nrow(meta_old) - nrow(meta)} station{?s} \\
                      with missing latitude & longitude."
        )
      }

      meta_sf <- sf::st_as_sf(
        meta,
        coords = c("longitude", "latitude"),
        crs = 4326,
        remove = FALSE
      )

      meta_sf$distance_km <- as.numeric(sf::st_distance(meta_sf, target)) / 1000

                                          
      meta <- dplyr::arrange(meta_sf, .data$distance_km) |>
        sf::st_drop_geometry()

                           
      if (!is.null(max_dist)) {
        meta <- dplyr::filter(meta, .data$distance_km <= max_dist)
        if (nrow(meta) == 0) {
          warn_empty_meta()
          return(meta)
        }
      }

                         
      if (!is.null(max_n)) {
                                                                               
                
        max_n_sites <-
          meta |>
          dplyr::distinct(
            .data[["source"]],
            .data[["code"]],
            .data[["distance_km"]]
          ) |>
          dplyr::slice_min(
            order_by = .data$distance_km,
            n = max_n,
            with_ties = TRUE
          )

        meta <- dplyr::semi_join(meta, max_n_sites, by = c("source", "code"))
      }
    }

    dplyr::tibble(meta)
  }

                               
read_wsp_meta <- function(url, all, year, pollutant) {
  tmp <- tempfile()

             
  con <- url(url)
  meta <- get(load(con))
  close(con)

                   
  meta <- dplyr::rename(
    .data = meta,
    code = "site_id",
    site = "site_name",
    site_type = "location_type",
    variable = "parameter"
  ) |>
    dplyr::mutate(
      start_date = lubridate::ymd(.data$start_date, tz = "GMT"),
      site_type = gsub(
        pattern = "Urban traffic",
        replacement = "Urban Traffic",
        .data$site_type
      ),
      site_type = gsub(
        pattern = "unknown unknown",
        replacement = "Unknown",
        .data$site_type
      )
    )

                                                  
  if ("ratified_to" %in% names(meta)) {
    meta$ratified_to <-
      lubridate::ymd(
        meta$ratified_to,
        tz = lubridate::tz(meta$start_date),
        quiet = TRUE
      )
  }

                                                   
  id <- which(meta$end_date == "ongoing")
  meta$end_date[id] <- as.character(Sys.Date())
  meta$end_date <- lubridate::as_datetime(
    meta$end_date,
    tz = lubridate::tz(meta$start_date)
  )

                                              
  if (!is.null(year)) {
    meta$end_year <- lubridate::year(meta$end_date)
    meta$end_date[id] <- NA
    meta$start_year <- lubridate::year(meta$start_date)
    meta <-
      dplyr::filter(
        meta,
        .data$start_year <= min(year) &
          .data$end_year >= max(year)
      ) |>
      dplyr::select(
        -"end_year",
        -"start_year"
      )
  } else {
    meta$end_date[id] <- NA
  }

                              
  meta$variable[meta$variable == "NOXasNO2"] <- "NOx"

                        
  if (!is.null(pollutant)) {
    if ("hc" %in% pollutant) {
      pollutant <- c(
        pollutant,
        c(
          "123TMB",
          "124TMB",
          "135TMB",
          "13BDIENE",
          "1BUTENE",
          "1PENTEN",
          "2MEPENT",
          "3MEPENT",
          "ACETALDEHYDE",
          "ACETONE",
          "BENZENE",
          "c2BUTENE",
          "c2PENTEN",
          "CYCLOPENTANE",
          "ETHANE",
          "ETHANOL",
          "ETHBENZ",
          "ETHENE",
          "ETHYNE",
          "iBUTANE",
          "iBUTENE",
          "iOCTANE",
          "iPENTANE",
          "ISOPRENE",
          "MEPENT",
          "mpXYLENE",
          "mXYLENE",
          "nBUTANE",
          "nHEPTANE",
          "nHEXANE",
          "nNONANE",
          "nOCTANE",
          "nPENTANE",
          "oXYLENE",
          "PROPANE",
          "PROPENE",
          "pXYLENE",
          "t2BUTENE",
          "t2PENTEN",
          "TOLUENE"
        )
      )
      pollutant <- pollutant[pollutant != "hc"]
    }

    meta <- dplyr::filter(
      meta,
      tolower(.data$variable) %in% tolower({{ pollutant }})
    )
  }

                                                                
                                    
  if (!all) {
    meta <- dplyr::distinct(meta, .data$site, .keep_all = TRUE)
  }

  return(meta)
}

                                            
read_imperial_meta <- function(url, all, year) {
  con <- url(url)
  meta <- get(load(con))
  close(con)

                                                  
  meta <- dplyr::rename(
    meta,
    code = "SiteCode",
    site = "SiteName",
    site_type = "Classification",
    latitude = "Latitude",
    longitude = "Longitude"
  )

                                              
  if (!is.null(year)) {
                                                     
    meta$end_year <-
      lubridate::year(as.Date(meta$ClosingDate))
    meta$end_year <-
      ifelse(
        is.na(meta$end_year),
        as.numeric(format(Sys.Date(), "%Y")),
        meta$end_year
      )
    meta$start_year <- lubridate::year(meta$OpeningDate)
    meta <-
      dplyr::filter(
        meta,
        .data$start_year <= min(year) &
          .data$end_year >= max(year)
      )
  }

  meta
}

                                     
read_saqgetr_meta <- function(url, year) {
  msg <-
    c(
      "!" = "{.fun importEurope} has been discontinued and cannot import \\
      data after February 2024.",
      "i" = "Consider using the EEA Air Quality Download Service \\
      instead {.url https://eeadmz1-downloads-webapp.azurewebsites.net/}"
    )

  cli::cli_inform(msg, .frequency = "regularly", .frequency_id = "europe")

                     
  col_types <- readr::cols(
    site = readr::col_character(),
    site_name = readr::col_character(),
    latitude = readr::col_double(),
    longitude = readr::col_double(),
    elevation = readr::col_double(),
    country = readr::col_character(),
    country_iso_code = readr::col_character(),
    site_type = readr::col_character(),
    site_area = readr::col_character(),
    date_start = readr::col_character(),
    date_end = readr::col_character(),
    network = readr::col_character(),
    eu_code = readr::col_character(),
    eoi_code = readr::col_character(),
    data_source = readr::col_character()
  )

                             
  meta <-
    readr::read_csv(url, col_types = col_types, progress = FALSE) |>
    dplyr::mutate(
      date_start = lubridate::ymd_hms(.data$date_start, tz = "UTC"),
      date_end = lubridate::ymd_hms(.data$date_end, tz = "UTC")
    )

                                              
  if (!is.null(year)) {
                                                     
    meta$end_year <-
      lubridate::year(as.Date(meta$date_end))
    meta$end_year <-
      ifelse(
        is.na(meta$end_year),
        as.numeric(format(Sys.Date(), "%Y")),
        meta$end_year
      )
    meta$start_year <- lubridate::year(meta$date_start)
    meta <-
      dplyr::filter(
        meta,
        .data$start_year <= min(year) &
          .data$end_year >= max(year)
      )
  }

                         
  meta <- dplyr::rename(meta, code = "site", site = "site_name")

  meta
}

                                                           
warn_empty_meta <- function() {
  cli::cli_warn(
    c(
      "!" = "No metadata has been returned.",
      "i" = "Check the {.arg year}, {.arg code}, {.arg site}, \\
      {.arg pollutant} and/or {.arg max_dist} arguments."
    )
  )
}
