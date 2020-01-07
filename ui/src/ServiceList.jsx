import React from 'react'
import Grid from '@material-ui/core/Grid'
import Tabs from '@material-ui/core/Tabs'
import Tab from '@material-ui/core/Tab'

import { getServices, callServices } from './api'
import ServiceLine from './ServiceLine'
import Text from './Text'
import Response from './Response'

function intersection(xs, ys) {
  let result = []
  xs.forEach((x) => { if (ys.some((y) => y === x)) result.push(x) })
  return result
}

class ServiceList extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      text: '',
      selected: {},
      extraParams: {},
      json: {},
      debug: false,
      lang: undefined,
      response: undefined,
      visibleTab: 0
    }

    getServices
      .then((response) => response.json())
      .then((json) => this.setState({
        json: json.services
      }))
  }

  onTextChange(event) {
    this.setState({ text: event.target.value })
  }

  onSelect(task, name, langs, extraParams) {
    if (this.state.json[task]) {
      let currentSelection = this.state.selected
      if (currentSelection[task] && (currentSelection[task]['name'] === name)) {
        currentSelection[task] = undefined
      }
      else {
        currentSelection[task] = {
          name: name,
          langs: langs,
          extraParams: extraParams
        }
      }
      this.setState({ selected: currentSelection })
    }
  }

  onSend() {
    let tasks = []
    Object.entries(this.state.selected).forEach(([k, v]) => {
      tasks.push({task: k, name: v.name})
    })
    let request = {
      text: this.state.text,
      tasks: tasks,
      debug: this.state.debug
    }
    if (this.state.lang) {
      request.lang = this.state.lang
    }
    Object.entries(this.state.extraParams).forEach(([k, v]) => {
      request[k] = v
    })
    callServices(request)
      .then((response) => response.json())
      .then((json) => this.onResults(json))
  }

  onToggleDebug() {
    this.setState({ debug: !this.state.debug })
  }

  onResults(response) {
    this.setState({
      response: response,
      visibleTab: 1
    })
  }

  onChangeTab(event, tab) {
    this.setState({ visibleTab: tab })
  }

  onChooseLang(event, target) {
    const lang = (target.key === 'none') ? undefined : target.key
    this.setState({ lang: lang })
  }

  onChangeParam(param, value) {
    let currentParams = this.state.extraParams
    currentParams[param] = value
    this.setState({ extraParams: currentParams })
  }

  collectLangs() {
    var result = null
    Object.entries(this.state.selected).forEach(([k, v]) => {
      var langs = []
      if (v && v.langs) {
        v.langs.forEach((lang) => {
          if (lang !== '*') {
            langs.push(lang)
          }
        })
      }
      if (langs.length > 0) {
        if (result == null) {
          result = langs
        }
        else {
          result = intersection(result, langs)
        }
      }
    })

    return result
  }

  collectExtraParams() {
    var result = []
    Object.entries(this.state.selected).forEach(([k, v]) => {
      if (v && v.extraParams) {
        result = result.concat(v.extraParams)
      }
    })
    return result
  }

  render() {
    const onSelect = this.onSelect.bind(this)
    const lowerView = (this.state.visibleTab === 0) ?
      Object.entries(this.state.json).map(
        ([k, v]) => <ServiceLine
          key={k}
          task={k}
          implementations={v}
          selected={this.state.selected[k] && this.state.selected[k]['name']}
          onSelect={onSelect}
        />
      ) :
      <Response response={this.state.response} />

    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Text
            onChange={this.onTextChange.bind(this)}
            content={this.state.text}
            debug={this.state.debug}
            onToggleDebug={this.onToggleDebug.bind(this)}
            onSend={this.onSend.bind(this)}
            lang={this.state.lang}
            langs={this.collectLangs()}
            onChooseLang={this.onChooseLang.bind(this)}
            extraParams={this.collectExtraParams()}
            onChangeParam={this.onChangeParam.bind(this)}
          />
        </Grid>
        <Grid item xs={12}>
          <Tabs
            value={this.state.visibleTab}
            onChange={this.onChangeTab.bind(this)}
            indicatorColor='primary'
            textColor='primary'
            variant='fullWidth'
          >
            <Tab label='Services' />
            <Tab label='Response' />
          </Tabs>
        </Grid>
        {lowerView}

      </Grid>
    )
  }
}

export default ServiceList