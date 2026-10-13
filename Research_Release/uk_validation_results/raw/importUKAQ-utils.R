                                                                          
        
readUKAQData <-
  function(
    site = "my1",
    year = 2009,
    data_type = "hourly",
    pollutant = "all",
    hc = FALSE,
    ratified = FALSE,
    to_narrow = FALSE,
    verbose = FALSE,
    source = "aurn",
    url_data,
    lmam_subfolder
  ) {
                                     
    if (source == "local") {
      url_data <- paste0(url_data, lmam_subfolder, "/")
    }

                                               
    site <- toupper(site)

    files <- paste0(site, "_", year)

                             
    thedata <-
      suppressWarnings(loadData(
        x = files,
        verbose = verbose,
        url_data = url_data,
        data_type = data_type
      ))

                                                           
    if (is.null(thedata)) {
      cli::cli_abort(
        "No data to import for {.arg site} {.field {site}} and {.arg year} {.field {year}} from {.arg source} {.field {source}}."
      )
    }

                       
    if (nrow(thedata) == 0) {
      return()
    }           

                  
    names(thedata) <- tolower(names(thedata))

                       
    id <- which(names(thedata) %in% "noxasno2")
    if (length(id) == 1) {
      names(thedata)[id] <- "nox"
    }

                              
    thedata$code <- as.character(thedata$code)

                                      
    if (!hc) {
                                                                   
      theNames <- c(
        "site",
        "code",
        "date",
        "co",
        "nox",
        "no2",
        "no",
        "o3",
        "so2",
        "pm10",
        "pm2.5",
        "v10",
        "v2.5",
        "nv10",
        "nv2.5",
        "gr_pm10",
        "gr_pm2.5",
        "ws",
        "wd",
        "temp"
      )

      thedata <- dplyr::select(
        thedata,
        dplyr::any_of(theNames),
        dplyr::matches("_qc")
      )
    } else {
      thedata <- dplyr::relocate(
        thedata,
        dplyr::any_of(c("site", "code", "date"))
      )
    }

                                                
    if ("temp" %in% names(thedata)) {
      thedata <- dplyr::rename(thedata, air_temp = "temp")
    }

                                                 
    if (pollutant[1] != "all") {
      thedata <-
        dplyr::select(
          thedata,
          "date",
          "site",
          "code",
          dplyr::any_of(c(pollutant, "ws", "wd", "air_temp"))
        )
    }

                            
    attr(thedata$date, "tzone") <- "GMT"

                            
    if (to_narrow) {
                                         
      the_vars <- c(
        "date",
        "site",
        "code",
        "latitude",
        "longitude",
        "site_type",
        "ws",
        "wd",
        "air_temp"
      )

      thedata <-
        tidyr::pivot_longer(
          thedata,
          cols = -dplyr::any_of(the_vars),
          names_to = "pollutant"
        )

                                          
      thedata <- thedata |>
        dplyr::relocate(dplyr::any_of(the_vars)) |>
        dplyr::arrange(.data$site, .data$code, .data$pollutant, .data$date)
    }

                          
    thedata <-
      dplyr::mutate(thedata, source = source, .before = dplyr::everything())

    dplyr::as_tibble(thedata)
  }

                                          
        
loadData <- function(x, verbose, url_data, data_type) {
  tryCatch(
    {
                           
      fileName <- paste0(
        url_data,
        x,
        ".RData"
      )

                             
      con <- url(fileName)
      load(con)

                                                  
      x <- switch(
        data_type,
        hourly = x,
        `15min` = paste0(x, "_15min"),
        daily = paste0(x, "_daily_mean"),
        `8_hour` = paste0(x, "_8hour_mean"),
        `24_hour` = paste0(x, "_24hour_mean"),
        daily_max_8 = paste0(x, "_daily_max_8hour")
      )

                                          
                                                                              
      x2 <- gsub("_daily_mean", "_daily", x)

               
      dat <- get(x)

                                                     
      if (data_type == "daily" && exists(x2)) {
        dat2 <- get(x2)
        dat <- dplyr::left_join(dat, dat2, by = c("date", "site", "code"))

        lookup <- c(gr_pm2.5 = "GR2.5", gr_pm10 = "GR10")

        dat <- dat |>
          dplyr::rename(dplyr::any_of(lookup))
      }

                                                
      class(dat$date) <- c("POSIXct", "POSIXt")

      return(dat)
    },
    error = function(ex) {
                       
      if (verbose) {
        cli::cli_warn(c("i" = "{x} does not exist - ignoring that one."))
        return(NULL)
      }
    },
    finally = {
      close(con)
    }
  )
}

                                           
        
readSummaryData <-
  function(files, year, source, data_type, to_narrow, meta, hc) {
    fileName <- purrr::map(year, function(x) paste0(files, x, ".rds")) |>
      purrr::list_c()

    con <- url(fileName)
    on.exit(close.connection(con))
    thedata <- try(readRDS(con), TRUE)

    if (inherits(thedata, "try-error")) {
      return()
    }

    names(thedata)[names(thedata) == "NOXasNO2.mean"] <- "nox"
    names(thedata)[names(thedata) == "NOXasNO2.capture"] <-
      "nox_capture"
    names(thedata) <- tolower(names(thedata))
    names(thedata) <- gsub(".mean", "", names(thedata))
    names(thedata) <- gsub(".capture", "_capture", names(thedata))

    if (!hc) {
      thedata <- thedata |>
        dplyr::select(dplyr::any_of(
          c(
            "date",
            "uka_code",
            "code",
            "site",
            "year",
            "o3",
            "o3_capture",
            "o3.summer_capture",
            "o3.daily.max.8hour",
            "o3.aot40v",
            "o3.aot40f",
            "somo35",
            "somo35_capture",
            "no",
            "no_capture",
            "no2",
            "no2_capture",
            "nox",
            "nox_capture",
            "so2",
            "so2_capture",
            "co",
            "co_capture",
            "pm10",
            "pm10_capture",
            "nv10",
            "nv10_capture",
            "v10",
            "v10_capture",
            "pm2.5",
            "pm2.5_capture",
            "nv2.5",
            "nv2.5_capture",
            "v2.5",
            "v2.5_capture",
            "gr10",
            "gr10_capture",
            "gr2.5",
            "gr2.5_capture",
            "gr_pm2.5",
            "gr_pm10"
          )
        ))
    }

    if (data_type == "monthly") {
      thedata <- dplyr::mutate(thedata, date = lubridate::ymd(date, tz = "UTC"))
    }

    if (data_type == "annual") {
      thedata <- dplyr::rename(thedata, date = year) |>
        tidyr::drop_na(date) |>
        dplyr::mutate(date = lubridate::ymd(paste0(date, "-01-01"), tz = "UTC"))
    }

    if (to_narrow) {
                                     
      values <- dplyr::select(thedata, !dplyr::contains("capture")) |>
        dplyr::select(!dplyr::matches("uka_code"))

      capture <-
        dplyr::select(
          thedata,
          dplyr::contains("capture") | c("code", "date", "site")
        ) |>
        dplyr::select(!dplyr::matches("uka_code"))

      values <- tidyr::pivot_longer(
        values,
        -c("date", "code", "site"),
        values_to = "value",
        names_to = "species"
      )

      capture <- tidyr::pivot_longer(
        capture,
        -c("date", "code", "site"),
        values_to = "data_capture",
        names_to = "species"
      )

      capture$species <- gsub("_capture", "", capture$species)

      thedata <- dplyr::full_join(
        values,
        capture,
        by = c("date", "code", "site", "species")
      )
    }

    thedata <- thedata |>
      dplyr::mutate(
        site = as.character(.data$site),
        code = as.character(.data$code)
      ) |>
      dplyr::mutate(
        source = source,
        .before = dplyr::everything()
      )

    return(thedata)
  }

                                         
        
readDAQI <- function(files, year, source) {
  fileName <- purrr::map(year, function(x) paste0(files, x, ".rds")) |>
    purrr::list_c()

  con <- url(fileName)
  on.exit(close.connection(con))
  thedata <- try(readRDS(con), TRUE)

  if (inherits(thedata, "try-error")) {
    return()
  }

  thedata <- thedata |>
    dplyr::mutate(
      code = as.character(.data$code),
      site = as.character(.data$site),
      pollutant = as.character(.data$pollutant),
      date = lubridate::ymd(.data$Date, tz = "GMT"),
      measurement_period = as.character(.data$measurement_period)
    ) |>
    dplyr::select(-"Date") |>
    dplyr::relocate("date", .after = "pollutant") |>
    dplyr::mutate(
      source = source,
      .before = dplyr::everything()
    )

  return(thedata)
}

                                                                   
                                                   
                                                                        
        
add_meta <- function(source, columns, aq_data) {
  meta_data <-
    importMeta(
      source = unique(source),
      all = TRUE,
      duplicate = TRUE
    )

  meta_data <- dplyr::distinct(
    meta_data,
    .data[["source"]],
    .data[["site"]],
    .keep_all = TRUE
  ) |>
    dplyr::select("source", "site", "code", dplyr::all_of(columns))

  aq_data <- dplyr::left_join(
    aq_data,
    meta_data,
    by = c("source", "code", "site")
  )

  return(aq_data)
}

                                               
        
add_ratified <- function(aq_data, source, to_narrow) {
  meta <-
    importMeta(unique(source), all = TRUE) |>
    dplyr::filter(
      .data$code %in% aq_data$code,
      !.data$variable %in%
        c(
          "V10",
          "NV10",
          "V2.5",
          "NV2.5",
          "ws",
          "wd",
          "temp"
        )
    ) |>
    dplyr::select("source", "code", "variable", "ratified_to") |>
    dplyr::mutate(variable = tolower(.data$variable))

  if (to_narrow) {
    meta <-
      dplyr::rename(meta, "pollutant" = "variable", "qc" = "ratified_to") |>
      dplyr::filter(.data$pollutant %in% tolower(aq_data$pollutant))

    aq_data <-
      aq_data |>
      dplyr::left_join(
        meta,
        by = dplyr::join_by("source", "code", "pollutant")
      ) |>
      dplyr::mutate(qc = lubridate::floor_date(.data$date, "day") <= .data$qc)

    return(aq_data)
  }

  meta <-
    meta |>
    dplyr::filter(.data$variable %in% names(aq_data)) |>
    tidyr::pivot_wider(
      names_from = "variable",
      values_from = "ratified_to",
      names_glue = "{variable}_qc"
    )

  aq_data <-
    aq_data |>
    dplyr::left_join(meta, by = dplyr::join_by("source", "code")) |>
    dplyr::mutate(dplyr::across(dplyr::contains("_qc"), function(x) {
      lubridate::floor_date(.data$date, "day") <= x
    }))

  return(aq_data)
}


                                             
                                                                
        
filter_site_pollutant <- function(
  aq_data,
  site,
  pollutant,
  to_narrow,
  data_type
) {
                                          
  if (any(site != "all")) {
    aq_data <- aq_data[tolower(aq_data$code) %in% tolower(site), ]
  }

                                               
  if (any(pollutant != "all")) {
    polls <- paste(
      c("source", "uka_code", "code", "site", "date", "pollutant", pollutant),
      collapse = "|"
    )
    if (data_type != "daqi") {
      if (to_narrow) {
        aq_data <- aq_data[grepl(polls, aq_data$species, ignore.case = TRUE), ]
      } else {
        aq_data <- aq_data[grepl(polls, names(aq_data), ignore.case = TRUE)]
      }
    } else {
      aq_data <- aq_data[grepl(polls, aq_data$pollutant, ignore.case = TRUE), ]
    }
  }

                 
  return(aq_data)
}

                                                   
                                             
        
guess_source <- function(site) {
  ukaq_meta <- importMeta("ukaq") |>
    dplyr::mutate(
      source = factor(
        .data$source,
        c("aurn", "saqn", "aqe", "waqn", "ni", "local")
      ),
      code = toupper(.data$code)
    ) |>
    dplyr::arrange(.data$source) |>
    dplyr::distinct(
      .data$site,
      .data$code,
      .data$latitude,
      .data$longitude,
      .keep_all = TRUE
    )

  source_tbl <-
    data.frame(code = toupper(site)) |>
    dplyr::left_join(ukaq_meta, by = "code")

  if (anyNA(source_tbl$source)) {
    ambiguous_codes <-
      source_tbl |>
      dplyr::filter(is.na(.data$source)) |>
      dplyr::pull(.data$code)

    cli::cli_abort(
      c(
        "x" = "Unknown site codes detected. Please ensure all site codes can be found in {.fun importMeta}.",
        "i" = "Unknown site codes: {ambiguous_codes}"
      )
    )
  }

  if (nrow(source_tbl) > length(site)) {
    source_tbl_all <- source_tbl

    source_tbl <- dplyr::slice_head(source_tbl, n = 1, by = "code")

    source_tbl_other <-
      dplyr::anti_join(
        source_tbl_all,
        source_tbl,
        by = dplyr::join_by(
          "code",
          "source",
          "site",
          "latitude",
          "longitude",
          "site_type"
        )
      ) |>
      dplyr::distinct()

    alternatives <-
      source_tbl_other |>
      dplyr::mutate(
        str = paste0(
          .data$code,
          " (could also be '",
          .data$site,
          "' from the source: '",
          .data$source,
          "'.)"
        )
      ) |>
      dplyr::pull(.data$str)

    msg <- c(
      "x" = "Ambiguous site codes detected. National networks are imported preferentially to locally managed networks.",
      alternatives,
      "i" = "Specify {.field source} to import sites from specific monitoring networks."
    )

    names(msg)[names(msg) == ""] <- "!"

    cli::cli_warn(msg)
  }

  return(source_tbl$source)
}
